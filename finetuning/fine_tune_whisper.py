import torch
from dataclasses import dataclass
from typing import Any, Dict, List

from datasets import load_dataset, DatasetDict, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate
from peft import LoraConfig, TaskType, get_peft_model


def load_dataset_splits(
    dataset_name: str,
    subset: str = "default",
    sampling_rate: int = 16000,
    val_split: float = 0.1,
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
    features = processor.feature_extractor(audio_arrays, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="np")
    labels = processor.tokenizer(batch["transcript"], return_tensors="np", padding=True).input_ids
    batch["input_features"] = features.input_features
    batch["labels"] = labels
    return batch

@dataclass
class DataCollatorSpeechWithPadding:
    processor: WhisperProcessor

    def __call__(
        self, features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        audio_inputs = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(audio_inputs, return_tensors="pt")

        label_inputs = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_inputs, return_tensors="pt")
        labels = labels_batch.input_ids
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch


def compute_metrics(pred, processor: WhisperProcessor) -> Dict[str, float]:
    wer_metric = evaluate.load("wer")
    preds = processor.tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
    labels = pred.label_ids
    labels[labels == -100] = processor.tokenizer.pad_token_id
    refs = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
    wer = wer_metric.compute(predictions=preds, references=refs)
    return {"wer": wer * 100}


def main():
    model_name = "openai/whisper-large-v3"
    processor = WhisperProcessor.from_pretrained(model_name, language="en", task="transcribe")

    dataset = load_dataset_splits("pwmm/LS-dev-other-mbr-vocals-fv4-gabox")
    dataset = dataset.map(
        lambda batch: preprocess_batch(batch, processor),
        remove_columns=dataset["train"].column_names,
        batched=True,
        load_from_cache_file=True,
    )

    base_model = WhisperForConditionalGeneration.from_pretrained(model_name)
    base_model.config.update({"language": "en", "task": "transcribe"})

    lora_config = LoraConfig(
        inference_mode=False,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)

    data_collator = DataCollatorSpeechWithPadding(processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-lora",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        warmup_steps=500,
        max_steps=1000,
        gradient_checkpointing=False,
        max_grad_norm=1.0,
        fp16=True,
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=3,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        processing_class=processor,
        compute_metrics=lambda pred: compute_metrics(pred, processor),
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
