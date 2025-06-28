import argparse
import os
import mlflow
from config import DEFAULT_SMOGY_DIR, DEFAULT_TRAINED_MODELS_DIR, DEFAULT_DATASET_PATH_IMAGE
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
        default=DEFAULT_DATASET_PATH_IMAGE,
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
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="Number of training samples to use",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=None,
        help="Number of validation samples to use",
    )
    parser.add_argument(
        "--mlflow-uri",
        default="http://localhost:5000",
        help="Tracking URI for MLflow server",
    )
    parser.add_argument("--mlflow-experiment", default="authentiscore", help="MLflow experiment name")
    args = parser.parse_args()

    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.mlflow_experiment)

    dataset = load_dataset("imagefolder", data_dir=args.dataset_path)

    if args.train_size is not None:
        train_count = min(args.train_size, len(dataset["train"]))
        dataset["train"] = dataset["train"].shuffle(seed=42).select(range(train_count))
    if args.val_size is not None and "validation" in dataset:
        val_count = min(args.val_size, len(dataset["validation"]))
        dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(val_count))

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(final_output_dir, exist_ok=True)

    metrics = None
    if args.mlflow_uri:
        with mlflow.start_run(run_name="train_smogy"):
            mlflow.log_param("dataset_path", args.dataset_path)
            mlflow.log_param("model_dir", args.model_dir)
            trainer.train()
            if eval_ds is not None:
                metrics = trainer.evaluate()
                mlflow.log_metrics(metrics)
            trainer.save_model(final_output_dir)
            processor.save_pretrained(final_output_dir)
            mlflow.log_artifacts(final_output_dir)
    else:
        trainer.train()
        if eval_ds is not None:
            metrics = trainer.evaluate()
        trainer.save_model(final_output_dir)
        processor.save_pretrained(final_output_dir)
    if metrics is not None:
        print(f"Evaluation: {metrics}")


if __name__ == "__main__":
    main()
