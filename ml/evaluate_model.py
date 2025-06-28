import argparse
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate
from config import DEFAULT_DISTILBERT_DOWNLOAD_DIR, DEFAULT_DATASET_PATH_TEXT, USED_DATASET_PATH
import mlflow


class TextClassificationDataset(TorchDataset):
    """Dataset for on-the-fly tokenization."""

    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def load_dataset(
    path: str,
    sample_size: float | int | None = None,
    exclude_path: str | None = None,
) -> pd.DataFrame:
    """Load dataset and optionally sample a subset.

    If ``exclude_path`` is given, rows with matching ``text`` values in that
    dataset are removed to avoid evaluating on the training/validation data.
    """
    df = pd.read_csv(path)

    if "text" not in df.columns:
        raise ValueError("Dataset must have a 'text' column")
    if "label" not in df.columns:
        if "generated" in df.columns:
            df = df.rename(columns={"generated": "label"})
        elif "is_ai_generated" in df.columns:
            df = df.rename(columns={"is_ai_generated": "label"})
        else:
            raise ValueError("Dataset must have a 'label', 'generated', or 'is_ai_generated' column")

    df["label"] = df["label"].astype(float).astype(int)

    if exclude_path:
        exclude_df = pd.read_csv(exclude_path)
        if "text" not in exclude_df.columns:
            raise ValueError("Exclude dataset must have a 'text' column")
        df = df[~df["text"].isin(exclude_df["text"])]
        df = df.reset_index(drop=True)

    if sample_size is not None:
        if isinstance(sample_size, float) and 0 < sample_size < 1:
            df = df.sample(frac=sample_size, random_state=42)
        else:
            df = df.sample(n=min(int(sample_size), len(df)), random_state=42)
        df = df.reset_index(drop=True)

    return df


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    precision = precision_metric.compute(predictions=predictions, references=labels, average="weighted")
    recall = recall_metric.compute(predictions=predictions, references=labels, average="weighted")
    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
        "precision": precision["precision"],
        "recall": recall["recall"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned text classification model")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH_TEXT, help="Path to CSV dataset")
    parser.add_argument("--model-dir", default=DEFAULT_DISTILBERT_DOWNLOAD_DIR, help="Path to fine-tuned model directory")
    parser.add_argument("--sample-size", type=float, default=400, help="Number of samples or fraction to evaluate")
    parser.add_argument("--exclude-path", default=USED_DATASET_PATH, help="CSV file with training/validation texts to exclude")
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

    df = load_dataset(args.dataset_path, args.sample_size, args.exclude_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    max_length = 256
    eval_dataset = TextClassificationDataset(df["text"].tolist(), df["label"].tolist(), tokenizer, max_length)

    device_str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    use_fp16 = device_str == "cuda"
    training_args = TrainingArguments(
        output_dir=os.path.join(args.model_dir, "eval"),
        per_device_eval_batch_size=8,
        fp16=use_fp16,
        no_cuda=device_str != "cuda",
        dataloader_drop_last=False,
        report_to="none",
    )

    trainer = Trainer(model=model, args=training_args, eval_dataset=eval_dataset, compute_metrics=compute_metrics)

    if args.mlflow_uri:
        with mlflow.start_run(run_name="evaluate_model"):
            mlflow.log_param("dataset_path", args.dataset_path)
            mlflow.log_param("model_dir", args.model_dir)
            metrics = trainer.evaluate()
            mlflow.log_metrics(metrics)
    else:
        metrics = trainer.evaluate()
    print(f"Evaluation results: {metrics}")


if __name__ == "__main__":
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    main()
