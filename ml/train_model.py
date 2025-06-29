import os
import argparse
import mlflow
from config import (
    DEFAULT_DATASET_PATH_TEXT,
    DEFAULT_MODEL_ID,
    DEFAULT_TRAINED_MODELS_DIR,
    DEFAULT_DISTILBERT_DOWNLOAD_DIR,
    USED_DATASET_PATH,
)
from datetime import datetime
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import evaluate


class TextClassificationDataset(TorchDataset):
    """Simple torch Dataset that tokenizes on‑the‑fly."""

    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length", max_length=max_length
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def load_and_prepare_dataset(path: str):
    """Loads a CSV with `text` + {label|generated|is_ai_generated} columns and balances it."""

    print(f"Loading dataset from {path}")
    df = pd.read_csv(path)

    # --- column sanity check / rename ------------------------------------------------
    if "text" not in df.columns:
        raise ValueError("Dataset must have a 'text' column")
    if "label" not in df.columns:
        if "generated" in df.columns:
            df = df.rename(columns={"generated": "label"})
        elif "is_ai_generated" in df.columns:
            df = df.rename(columns={"is_ai_generated": "label"})
        else:
            raise ValueError(
                "Dataset must have a 'label', 'generated', or 'is_ai_generated' column"
            )

    # --- make sure labels are 0/1 ints ----------------------------------------------
    df["label"] = df["label"].astype(float).astype(int)
    label_counts = df["label"].value_counts()

    # --- balance classes (max 2 000 each) --------------------------------------------
    samples_per_class = min(2000, label_counts.min())
    df_class_0 = df[df["label"] == 0].sample(samples_per_class, random_state=42)
    df_class_1 = df[df["label"] == 1].sample(samples_per_class, random_state=42)
    df = (
        pd.concat([df_class_0, df_class_1])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    print(f"Prepared balanced dataset with {len(df)} samples")
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    return train_df, val_df


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    precision = precision_metric.compute(
        predictions=predictions, references=labels, average="weighted"
    )
    recall = recall_metric.compute(
        predictions=predictions, references=labels, average="weighted"
    )
    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
        "precision": precision["precision"],
        "recall": recall["recall"],
    }


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train text‑classification model")
    parser.add_argument(
        "--dataset-path", default=DEFAULT_DATASET_PATH_TEXT, help="Path to CSV dataset"
    )
    parser.add_argument(
        "--model-name", default=DEFAULT_DISTILBERT_DOWNLOAD_DIR, help="HF model name"
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_TRAINED_MODELS_DIR, help="Where to save the model"
    )
    parser.add_argument(
        "--save-train-texts",
        default=USED_DATASET_PATH,
        help=(
            "Optional path to write a CSV with the texts used for"
            " training and validation."
        ),
    )
    parser.add_argument(
        "--mlflow-uri",
        default="http://localhost:5000",
        help="Tracking URI for MLflow server",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default="authentiscore",
        help="MLflow experiment name",
    )
    args = parser.parse_args()

    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.mlflow_experiment)

    if not args.dataset_path:
        raise ValueError("Dataset path must be provided via --dataset-path or env var")

    # -----------------------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------------------
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # -----------------------------------------------------------------------
    # Device selection: CUDA > MPS > CPU
    # -----------------------------------------------------------------------
    if torch.cuda.is_available():
        device_str = "cuda"
    elif torch.backends.mps.is_available():
        device_str = "mps"
    else:
        device_str = "cpu"
    print(f"Using device: {device_str}")

    use_fp16 = device_str == "cuda"          # only safe on Nvidia CUDA
    no_cuda_flag = device_str != "cuda"      # forces Trainer off CUDA (MPS/CPU)

    # -----------------------------------------------------------------------
    # Data prep
    # -----------------------------------------------------------------------
    train_df, val_df = load_and_prepare_dataset(args.dataset_path)

    if args.save_train_texts:
        used_df = pd.concat([train_df, val_df], ignore_index=True)
        used_df[["text", "label"]].to_csv(args.save_train_texts, index=False)
        print(f"Saved training texts to {args.save_train_texts}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    max_length = 256
    train_dataset = TextClassificationDataset(
        train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_length
    )
    val_dataset = TextClassificationDataset(
        val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, max_length
    )

    # -----------------------------------------------------------------------
    # Model + Trainer
    # -----------------------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        fp16=use_fp16,            # <- only on CUDA
        no_cuda=no_cuda_flag,     # <- allows MPS/CPU fallback
        logging_steps=10,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # -----------------------------------------------------------------------
    # Train & evaluate with optional MLflow logging
    # -----------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(final_output_dir, exist_ok=True)

    if args.mlflow_uri:
        with mlflow.start_run(run_name="train_model"):
            mlflow.log_param("dataset_path", args.dataset_path)
            mlflow.log_param("model_name", args.model_name)
            trainer.train()
            eval_results = trainer.evaluate()
            trainer.save_model(final_output_dir)
            tokenizer.save_pretrained(final_output_dir)
            mlflow.log_metrics(eval_results)
            mlflow.log_artifacts(final_output_dir)
    else:
        trainer.train()
        eval_results = trainer.evaluate()
        trainer.save_model(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)
    print(f"Model and tokenizer saved to {final_output_dir}")
    print(f"Evaluation results: {eval_results}")


if __name__ == "__main__":
    # Lazy‑load metrics (outside functions to avoid re‑download inside Trainer)
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    main()
