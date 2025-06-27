import argparse
import os
from config import DEFAULT_SMOGY_DIR, DEFAULT_TRAINED_MODELS_DIR
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)
import torch
from datetime import datetime


def preprocess(examples, processor):
    images = [img.convert("RGB") for img in examples["image"]]
    inputs = processor(images=images)
    examples["pixel_values"] = inputs["pixel_values"]
    return examples


def main():
    parser = argparse.ArgumentParser(description="Fine-tune the Smogy model")
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to an imagefolder dataset with train/val splits",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_SMOGY_DIR,
        help="Path to pretrained model directory",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_TRAINED_MODELS_DIR,
        help="Where to save the fine-tuned model",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    args = parser.parse_args()

    dataset = load_dataset("imagefolder", data_dir=args.dataset_path)

    processor = AutoImageProcessor.from_pretrained(args.model_dir)
    model = AutoModelForImageClassification.from_pretrained(args.model_dir)

    def transform(examples):
        return preprocess(examples, processor)

    dataset = dataset.map(transform, batched=True)
    dataset.set_format(type="torch", columns=["pixel_values", "label"])

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        logging_steps=10,
        save_total_limit=2,
        report_to="none",
    )

    eval_ds = dataset.get("validation") or None
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=eval_ds,
        tokenizer=processor,
    )

    trainer.train()
    if eval_ds is not None:
        metrics = trainer.evaluate()
        print(f"Evaluation: {metrics}")


    # Make output directory unique to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(final_output_dir, exist_ok=True)

    trainer.save_model(final_output_dir)
    processor.save_pretrained(final_output_dir)


if __name__ == "__main__":
    main()
