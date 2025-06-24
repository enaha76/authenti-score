import argparse
import onnx
import torch
from torchvision import models


def export_model(weights_path: str, output_path: str, img_size: int = 224, opset: int = 14) -> None:
    """Load a PyTorch image model and export it to ONNX format."""
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=opset,
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"Exported ONNX model to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export image model to ONNX")
    parser.add_argument("--weights", required=True, help="Path to trained .pth file")
    parser.add_argument("--output", default="image_model.onnx", help="Output ONNX file")
    parser.add_argument("--img-size", type=int, default=224, help="Image size used during training")
    parser.add_argument("--opset-version", type=int, default=14, help="ONNX opset version")
    args = parser.parse_args()
    export_model(args.weights, args.output, args.img_size, args.opset_version)

