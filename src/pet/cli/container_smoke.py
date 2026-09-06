from __future__ import annotations

import argparse
import json
import platform

import torch
import torchvision

from pet.models import PetModel, load_model_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the packaged model and requested PyTorch device with a forward pass."
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device; 'cuda' fails clearly when no GPU is available.",
    )
    parser.add_argument(
        "--model-manifest",
        default="resnet34-v1",
    )
    parser.add_argument("--image-size", type=int, default=32)
    return parser


def resolve_device(requested: str) -> torch.device:
    """Resolve a requested smoke-test device without silently masking GPU failures."""
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Check the NVIDIA driver, container toolkit, "
            "and that the container was started with GPU access."
        )
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def run_smoke(model_manifest: str, requested_device: str, image_size: int) -> dict[str, object]:
    """Construct the versioned model and execute both independently versioned heads."""
    if image_size < 32:
        raise ValueError("image-size must be at least 32 for the ResNet-34 backbone")
    device = resolve_device(requested_device)
    manifest = load_model_manifest(model_manifest)
    model = PetModel(manifest, pretrained_backbone=False).to(device).eval()
    image = torch.zeros(1, 3, image_size, image_size, device=device)
    with torch.inference_mode():
        classification = model(image, "classification")
        segmentation = model(image, "segmentation")
    result: dict[str, object] = {
        "status": "ok",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": str(device),
        "cuda_runtime": torch.version.cuda,
        "backbone_version": manifest.backbone.version,
        "classification_head_version": manifest.classification.version,
        "segmentation_head_version": manifest.segmentation.version,
        "classification_shape": list(classification.shape),
        "segmentation_shape": list(segmentation.shape),
    }
    if device.type == "cuda":
        result["gpu"] = torch.cuda.get_device_name(device)
    return result


def main() -> None:
    """Run the container smoke test and emit machine-readable diagnostics."""
    args = _parser().parse_args()
    result = run_smoke(args.model_manifest, args.device, args.image_size)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
