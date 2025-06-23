import argparse
import torch
import onnx
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def export_model(model_dir: str, output_path: str, opset: int = 14, max_length: int = 256) -> None:
    """Export a Hugging Face model to ONNX format and verify the result."""
    print(f"Loading model from {model_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model.eval()
    dummy = tokenizer(
        "Export to ONNX", return_tensors="pt", padding="max_length", truncation=True, max_length=max_length
    )

    input_ids = dummy["input_ids"]
    attention_mask = dummy["attention_mask"]

    print(f"Exporting model to {output_path}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size"},
            },
            opset_version=opset,
        )

    print("Verifying exported ONNX model")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified successfully")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Hugging Face model to ONNX")

    # Now optional with a default value
    parser.add_argument(
        "--model-dir",
        default="ml_models/trained_model",  # or your local default model path
        help="Path to the trained model directory (local or model name from Hugging Face hub)"
    )

    parser.add_argument(
        "--output-path",
        default="model.onnx",
        help="Where to save the exported ONNX model"
    )

    parser.add_argument(
        "--opset-version",
        type=int,
        default=14,
        help="ONNX opset version"
    )

    args = parser.parse_args()

    export_model(args.model_dir, args.output_path, args.opset_version)


if __name__ == "__main__":
    main()
