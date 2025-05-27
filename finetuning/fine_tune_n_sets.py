import argparse
import os
import torch
from dataclasses import dataclass
from typing import Any, Dict, List

from datasets import load_dataset, DatasetDict, Audio, concatenate_datasets
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
import evaluate
from accelerate import init_empty_weights, infer_auto_device_map
from peft import LoraConfig, get_peft_model


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper with LoRA on specified dataset splits.")
    parser.add_argument(
        "--num-epochs", "-e",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--set", "-s",
        type=str,
        choices=["ls", "atco", "atcod", "all", "ls-atcod"],
        default="all",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--gpu-id", "-g",
        type=int,
        default=0,
        help="CUDA GPU id to use (as in CUDA_VISIBLE_DEVICES)",
    )
    return parser.parse_args()


def load_dataset_splits(
    dataset_name: str,
    subset: str = "default",
    sampling_rate: int = 16000,
    val_split: float = 0.05,
) -> DatasetDict:
    raw = load_dataset(dataset_name, subset)
    if "validation" not in raw:
        split = raw["train"].train_test_split(test_size=val_split)
        raw = DatasetDict(train=split["train"], validation=split["test"])
    return raw.cast_column("audio", Audio(sampling_rate=sampling_rate))


def preprocess_batch(
    batch: Dict[str, Any],
    processor: WhisperProcessor,
) -> Dict[str, Any]:
    audio_arrays = [ex["array"] for ex in batch["audio"]]
    features = processor.feature_extractor(
        audio_arrays,
        sampling_rate=processor.feature_extractor.sampling_rate,
        return_tensors="np"
    )
    labels = processor.tokenizer(
        batch["transcript"],
        return_tensors="np",
        padding=True
    ).input_ids

    batch["input_features"] = features.input_features
    batch["labels"] = labels
    return batch


@dataclass
class DataCollatorSpeechWithPadding:
    processor: WhisperProcessor

    def __call__(
        self,
        features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        audio_inputs = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(audio_inputs, return_tensors="pt")
        batch["input_features"] = batch["input_features"].to(torch.float16)

        label_inputs = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_inputs, return_tensors="pt")
        labels = labels_batch.input_ids
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    NUM_EPOCH = args.num_epochs
    SET = args.set

    model_name = "openai/whisper-large-v3"
    processor = WhisperProcessor.from_pretrained(
        model_name,
        language="en",
        task="transcribe"
    )

    ds_ls = []
    ds_atcod = []
    ds_atco = []

    # load datasets
    if SET in ['ls', 'all', 'ls-atcod']:
        ds_ls = load_dataset_splits("pwmm/LS-dev-other-mbr-vocals-fv4-gabox")
    if SET in ['atcod', 'all','ls-atcod']:
        ds_atcod = load_dataset_splits("pwmm/atco2-mbr-vocals-fv4-gabox")
    if SET in ['atco', 'all']:
        ds_atco = load_dataset_splits("jlvdoorn/atco2-asr")
        ds_atco = ds_atco.rename_column('text', 'transcript')
        ds_atco = ds_atco.remove_columns(["info"])

    if SET == 'all':
        orig_train_sizes = {k: len(v["train"]) for k, v in [('ls', ds_ls), ('atco', ds_atco), ('atcod', ds_atcod)]}
        orig_val_sizes = {k: len(v["validation"]) for k, v in [('ls', ds_ls), ('atco', ds_atco), ('atcod', ds_atcod)]}

        print(f"Original train sizes: {orig_train_sizes}")
        print(f"Original validation sizes: {orig_val_sizes}")

        min_train = min(orig_train_sizes.values())
        min_val = min(orig_val_sizes.values())

        print(f"Resizing each train split to {min_train} examples")
        print(f"Resizing each validation split to {min_val} examples")

        train_samples = [
            split.shuffle(seed=42).select(range(min_train))
            for split in (ds_ls['train'], ds_atco['train'], ds_atcod['train'])
        ]
        val_samples = [
            split.shuffle(seed=42).select(range(min_val))
            for split in (ds_ls['validation'], ds_atco['validation'], ds_atcod['validation'])
        ]

        dataset = DatasetDict({
            "train": concatenate_datasets(train_samples),
            "validation": concatenate_datasets(val_samples),
        })

        print(f"Combined train size: {len(dataset['train'])} ({len(train_samples)} × {min_train})")
        print(f"Combined validation size: {len(dataset['validation'])} ({len(val_samples)} × {min_val})")

    elif SET == 'ls-atcod':
        dataset = DatasetDict({
            "train": concatenate_datasets([ds_ls["train"], ds_atcod["train"]]),
            "validation": concatenate_datasets([ds_ls["validation"], ds_atcod["validation"]]),
        })
    else:
        ds_map = {'ls': ds_ls, 'atco': ds_atco, 'atcod': ds_atcod}
        d = ds_map[SET]
        dataset = DatasetDict({
            "train": d["train"],
            "validation": d["validation"],
        })

    # preprocess
    dataset = dataset.map(
        lambda batch: preprocess_batch(batch, processor),
        remove_columns=dataset["train"].column_names,
        batched=True,
        load_from_cache_file=True,
    )

    # quantize & shard
    bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
    with init_empty_weights():
        dummy = WhisperForConditionalGeneration.from_pretrained(model_name)
    max_mem = {i: "24GiB" for i in range(torch.cuda.device_count())}
    device_map = infer_auto_device_map(dummy, max_memory=max_mem)

    base_model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
    )
    base_model.config.update({"language": "en", "task": "transcribe"})

    # apply LoRA
    lora_config = LoraConfig(
        inference_mode=False,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.tie_weights()

    data_collator = DataCollatorSpeechWithPadding(processor)

    # define compute_metrics
    def compute_metrics(eval_pred):
        wer_metric = evaluate.load("wer")
        pred_ids = eval_pred.predictions if not isinstance(eval_pred.predictions, tuple) else eval_pred.predictions[0]
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels = eval_pred.label_ids
        labels[labels == -100] = processor.tokenizer.pad_token_id
        ref_str = processor.batch_decode(labels, skip_special_tokens=True)
        wer = wer_metric.compute(predictions=pred_str, references=ref_str)
        return {"wer": wer}

    # training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"~/respro/finetuned_models/whisper-{SET}-{NUM_EPOCH}",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        num_train_epochs=NUM_EPOCH,
        gradient_checkpointing=False,
        max_grad_norm=1.0,
        optim="adafactor",
        bf16=True,
        eval_strategy="no",
        save_strategy="no",
        eval_steps=0.5,
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        logging_steps=10,
        save_total_limit=2,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=processor,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
