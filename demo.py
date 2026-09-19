from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import Tensor

from model import AUTHORIZED, UNAUTHORIZED, load_recognizer
from preprocessing import load_image, save_png
from protection import load_basis, protect


ROOT = Path(__file__).resolve().parent
MATCH_THRESHOLDS = {
    AUTHORIZED: 0.17313844710588455,
    UNAUTHORIZED: 0.1682187169790268,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-image IRSE-100 protection demo")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--basis", type=Path, default=ROOT / "assets" / "irse100_basis.pt")
    parser.add_argument("--weights-root", type=Path, default=ROOT / "external" / "weights")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.inference_mode()
def compare_embeddings(
    model: torch.nn.Module, clean: Tensor, protected: Tensor, reference: Tensor
) -> tuple[float, float]:
    embeddings = model(torch.cat((clean, protected, reference), dim=0))
    original_score = float((embeddings[0] * embeddings[2]).sum().item())
    protected_score = float((embeddings[1] * embeddings[2]).sum().item())
    return original_score, protected_score


def match_decision(score: float, threshold: float) -> str:
    return "MATCH" if score >= threshold else "NO MATCH"


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.reference is not None and args.image.resolve() == args.reference.resolve():
        raise ValueError("The reference must be a different image")
    device = torch.device(args.device)
    clean = load_image(args.image).to(device)
    reference = (
        load_image(args.reference).to(device)
        if args.reference is not None
        else None
    )
    basis = load_basis(args.basis)
    protected = protect(clean, basis, alpha=args.alpha, seed=args.seed)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    torch.save(protected.cpu(), output / "protected_tensor.pt")
    save_png(clean, output / "aligned_input.png")
    save_png(protected, output / "protected_preview.png")
    if reference is not None:
        save_png(reference, output / "reference_input.png")

    if reference is None:
        print("No match decision: supply --reference with a different face image.", flush=True)
    else:
        for role, name in (("authorized", AUTHORIZED), ("unauthorized", UNAUTHORIZED)):
            model = load_recognizer(name, args.weights_root, device)
            original_score, protected_score = compare_embeddings(
                model, clean, protected, reference
            )
            threshold = MATCH_THRESHOLDS[name]
            print(
                f"{role.capitalize()} ({name}): "
                f"original {match_decision(original_score, threshold)} "
                f"(similarity={original_score:.4f}) -> "
                f"protected {match_decision(protected_score, threshold)} "
                f"(similarity={protected_score:.4f})",
                flush=True,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    print(f"Saved outputs to {output}", flush=True)


if __name__ == "__main__":
    main()
