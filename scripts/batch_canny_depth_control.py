import argparse
from pathlib import Path
import random
import sys
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from huggingface_hub import snapshot_download

from restyle_image import ModelManager, run
from scene_transfer.config import RunConfig


HF_CHECKPOINTS = [
    "runwayml/stable-diffusion-v1-5",
    "lllyasviel/sd-controlnet-depth",
    "diffusers/controlnet-depth-sdxl-1.0",
    "madebyollin/sdxl-vae-fp16-fix",
    "stabilityai/stable-diffusion-xl-base-1.0",
]


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def normalize_format(fmt: str) -> str:
    fmt = fmt.strip().lower()
    if not fmt.startswith("."):
        fmt = "." + fmt
    return fmt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch ReStyle3D style transfer using restyle_image pipeline."
    )

    parser.add_argument(
        "--prompt",
        required=True,
        help=(
            "Prompt controlling the transferred "
            "style/content appearance."
        ),
    )

    parser.add_argument(
        "--scale",
        type=float,
        required=True,
        help=(
            "Direct swap guidance strength. "
            "This value is passed as swap_guidance_scale without remapping."
        ),
    )

    parser.add_argument(
        "-content_dir",
        "--content_dir",
        required=True,
        type=Path,
        help="Directory containing source images.",
    )

    parser.add_argument(
        "--format",
        required=True,
        help="Image extension to process, e.g. .jpeg, jpeg, .png.",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Directory receiving the generated images.",
    )

    parser.add_argument(
        "--style_image",
        type=Path,
        default=None,
        help=(
            "Reference style image used by IP-Adapter. "
            "If omitted, each input image is used as its own "
            "style image (self-style transfer)."
        ),
    )

    parser.add_argument(
        "--style_dir",
        type=Path,
        default=None,
        help=(
            "Optional directory of style images. If provided, one random style image "
            "is selected for each input image and this overrides --style_image."
        ),
    )

    parser.add_argument(
        "--name",
        type=str,
        default="ReStyle3D",
        help="Run name used for working directories.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=120,
        help="Number of diffusion steps (default: 120).",
    )

    parser.add_argument(
        "--skip_steps",
        type=int,
        default=32,
        help="Number of DDPM inversion steps to skip before generation (default: 32).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )

    parser.add_argument(
        "--load_latents",
        type=parse_bool,
        default=True,
        help="Load cached latents when available (true/false).",
    )

    parser.add_argument(
        "--filtering",
        type=parse_bool,
        default=True,
        help="Kept for CLI compatibility (not used by ReStyle3D pipeline).",
    )

    parser.add_argument(
        "--filter_perc",
        type=float,
        default=0.25,
        help="Kept for CLI compatibility (not used by ReStyle3D pipeline).",
    )

    parser.add_argument(
        "--adain_class",
        type=parse_bool,
        default=False,
        help="Kept for CLI compatibility (not used by ReStyle3D pipeline).",
    )

    parser.add_argument(
        "--domain_name",
        type=str,
        default="interior",
        help="Domain used by ReStyle3D refiner prompt (default: interior).",
    )

    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("--steps must be > 0.")

    if args.skip_steps < 0 or args.skip_steps >= args.steps:
        parser.error("--skip_steps must be >= 0 and < --steps.")

    if not 0.0 <= args.filter_perc <= 1.0:
        parser.error("--filter_perc must be in [0, 1].")

    if args.style_image is not None and not args.style_image.is_file():
        parser.error(f"Style image does not exist: {args.style_image}")

    if args.style_dir is not None and not args.style_dir.is_dir():
        parser.error(f"Style directory does not exist: {args.style_dir}")

    return args


def ensure_hf_checkpoints() -> None:
    print("Prefetching Hugging Face checkpoints for ReStyle3D...")
    for repo_id in HF_CHECKPOINTS:
        print(f"  - {repo_id}")
        snapshot_download(repo_id=repo_id)
    print("All Hugging Face checkpoints are available in local cache.")


def collect_content_images(content_dir: Path, image_format: str) -> List[Path]:
    input_files = sorted(
        path for path in content_dir.iterdir() if path.is_file() and path.suffix.lower() == image_format
    )
    if not input_files:
        raise SystemExit(f"No files with format '{image_format}' found in {content_dir}")
    return input_files


def resolve_struct_seg_dict(content_img: Path) -> Path:
    candidates = [
        content_img.with_suffix(".pth"),
        content_img.parent / "seg_dict" / f"{content_img.stem}.pth",
    ]
    if content_img.parent.name.lower() == "images":
        candidates.append(content_img.parent.parent / "seg_dict" / f"{content_img.stem}.pth")

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    candidate_list = "\n".join(str(path) for path in candidates)
    raise SystemExit(
        "Unable to resolve struct segmentation dictionary for content image "
        f"{content_img}. Tried:\n{candidate_list}"
    )


def resolve_style_seg_dict(style_img: Path) -> Optional[Path]:
    candidates = [
        style_img.parent / "seg_dict.pth",
        style_img.with_suffix(".pth"),
        style_img.parent / "seg_dict" / f"{style_img.stem}.pth",
    ]
    if style_img.parent.name.lower() == "images":
        candidates.append(style_img.parent.parent / "seg_dict" / f"{style_img.stem}.pth")

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def collect_style_images(style_dir: Path) -> List[Tuple[Path, Path]]:
    allowed_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    style_refs: List[Tuple[Path, Path]] = []

    for path in sorted(style_dir.iterdir()):
        if path.is_dir():
            image_candidates = sorted(path.glob("image.*"))
            seg_candidate = path / "seg_dict.pth"
            if image_candidates and seg_candidate.is_file():
                style_refs.append((image_candidates[0], seg_candidate))
            continue

        if path.is_file() and path.suffix.lower() in allowed_suffixes:
            seg_path = resolve_style_seg_dict(path)
            if seg_path is not None:
                style_refs.append((path, seg_path))

    if not style_refs:
        raise SystemExit(
            "No valid style references found in style_dir. "
            "Expected image files with corresponding seg dicts, or subfolders containing image.* and seg_dict.pth."
        )

    return style_refs


def make_cfg(
    args: argparse.Namespace,
    content_img: Path,
    style_img: Path,
    output_work_dir: Path,
    struct_seg: Path,
    style_seg: Path,
) -> RunConfig:
    cfg = RunConfig(
        app_image_path=style_img,
        struct_image_path=content_img,
        output_path=output_work_dir,
        domain_name=args.domain_name,
        seed=args.seed,
        prompt=args.prompt,
        num_timesteps=args.steps,
        load_latents=args.load_latents,
        skip_steps=args.skip_steps,
        swap_guidance_scale=args.scale,
        use_masked_adain=False,
    )
    cfg.config_exp()
    cfg.struct_seg_dict = str(struct_seg)
    cfg.app_seg_dict = str(style_seg)
    return cfg


def main() -> None:
    args = parse_args()

    content_dir = args.content_dir
    output_dir = args.output_dir
    image_format = normalize_format(args.format)

    if not content_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {content_dir}")

    ensure_hf_checkpoints()

    output_dir.mkdir(parents=True, exist_ok=True)
    work_root = output_dir / "_restyle3d_work"
    work_root.mkdir(parents=True, exist_ok=True)

    input_files = collect_content_images(content_dir, image_format)
    style_refs = collect_style_images(args.style_dir) if args.style_dir is not None else None

    if args.filtering is not True or args.filter_perc != 0.25 or args.adain_class:
        print("Warning: --filtering, --filter_perc, and --adain_class are ignored by ReStyle3D batch wrapper.")

    if style_refs is not None:
        print(f"Using random style reference per image from: {args.style_dir}")
        print(f"Discovered {len(style_refs)} style reference(s) with segmentation dictionaries.")
    elif args.style_image is None:
        print("No --style_image provided: using each input image as its own style reference.")
    else:
        print(f"Using global style reference: {args.style_image}")

    print(f"Found {len(input_files)} input file(s).")
    print(f"Output folder: {output_dir}")

    rng = random.Random(args.seed)
    pipelines: Optional[ModelManager] = None

    for index, input_path in enumerate(input_files, start=1):
        output_path = output_dir / input_path.name
        struct_seg = resolve_struct_seg_dict(input_path)

        if style_refs is not None:
            style_path, style_seg = rng.choice(style_refs)
        else:
            style_path = args.style_image if args.style_image is not None else input_path
            style_seg = resolve_style_seg_dict(style_path)
            if style_seg is None:
                raise SystemExit(
                    "Unable to resolve style segmentation dictionary for style image "
                    f"{style_path}. Expected one of: seg_dict.pth in parent folder, "
                    f"{style_path.stem}.pth next to image, or seg_dict/{style_path.stem}.pth."
                )

        work_dir = work_root / f"{index:05d}_{input_path.stem}" / "intermediate"

        print(
            f"[{index}/{len(input_files)}] "
            f"{input_path.name} style={style_path.name} -> {output_path}"
        )

        cfg = make_cfg(
            args=args,
            content_img=input_path,
            style_img=style_path,
            output_work_dir=work_dir,
            struct_seg=struct_seg,
            style_seg=style_seg,
        )

        if pipelines is None:
            pipelines = ModelManager(cfg)
        else:
            pipelines.transfer_pipe.config = cfg

        images = run(cfg, pipelines)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(output_path)

    print(f"Finished. Wrote {len(input_files)} file(s) to {output_dir}")


if __name__ == "__main__":
    main()