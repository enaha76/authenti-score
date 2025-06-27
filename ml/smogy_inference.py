import argparse
import os
from transformers import AutoImageProcessor, AutoModelForImageClassification
from config import DEFAULT_SMOGY_DIR
from PIL import Image
import torch
import numpy as np


def predict(image_path: str, model_dir: str) -> None:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir)

    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits.detach().cpu().numpy()
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    pred = int(np.argmax(probabilities, axis=1)[0])
    confidence = float(probabilities[0][pred])
    label = "AI-generated" if pred == 1 else "Real"
    print(f"Prediction: {label} (confidence {confidence:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Run inference with the Smogy model")
    parser.add_argument("image", help="Path to an image")
    parser.add_argument(
        "--model-dir", default=DEFAULT_SMOGY_DIR, help="Directory containing the model"
    )
    args = parser.parse_args()

    predict(args.image, args.model_dir)


if __name__ == "__main__":
    main()
