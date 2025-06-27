import os
import argparse
from config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_ID,
    DEFAULT_TEXT_MODEL_DIR,
)
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


def load_and_prepare_dataset(path):
    print(f"Loading dataset from {path}")
    df = pd.read_csv(path)
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
    df["label"] = df["label"].astype(float).astype(int)
    label_counts = df["label"].value_counts()
    # Balance the dataset with up to 500 samples per class
    samples_per_class = min(500, label_counts.min())
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


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(
        predictions=predictions, references=labels, average="weighted"
    )
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


def main():
    parser = argparse.ArgumentParser(description="Train text classification model")
    parser.add_argument(
        "--dataset-path",
        default=DEFAULT_DATASET_PATH,
        help="Path to CSV dataset",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_ID,
        help="Pretrained model name",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_TEXT_MODEL_DIR,
        help="Directory to save model",
    )
    args = parser.parse_args()

    if not args.dataset_path:
        raise ValueError(
            "Dataset path must be provided via --dataset-path or DATA_PATH env var"
        )

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    train_df, val_df = load_and_prepare_dataset(args.dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    max_length = 256
    train_dataset = TextClassificationDataset(
        train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_length
    )
    val_dataset = TextClassificationDataset(
        val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, max_length
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        fp16=torch.cuda.is_available(),
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

    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model and tokenizer saved to {args.output_dir}")


if __name__ == "__main__":
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    main()
