import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from tqdm import tqdm

from data_loader.dsec_full import DSECfull
from model.backbone_registry import BackboneRegistry
from training.config_loader import load_json
from training.metrics import compute_metrics
from training.trainer import resolve_device

METRIC_NAMES = ("raps", "ssim", "l1")
LOWER_IS_BETTER = {"raps": True, "ssim": False, "l1": True}
EPS = 1e-12

# Accumulated event grids hold real zeros — 81% of input voxels, 71% of ground truth —
# so their occupancy barely moves between thresholds 0 and 1e-2. A model output holds
# none, so any occupancy figure for it is set by the cutoff: 1.000 above 1e-6, 0.935
# above 1e-2, 0.489 above 0.1. Report the exact-zero fraction, which needs no cutoff,
# alongside occupancy at a threshold the caller can see and change. The default sits
# well under the median ground-truth event magnitude (0.64) and well over numerical
# dust, so it counts events rather than noise.
DEFAULT_OCCUPANCY_THRESHOLD = 0.1
STAT_NAMES = ("peak", "mean_abs", "zero_frac", "occupancy")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score a checkpoint against the input it refines, both measured "
                    "against ground truth"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a .pth checkpoint")
    parser.add_argument("--model_config", type=str,
                        help="Path to a JSON model config. Defaults to the config "
                             "stored inside the checkpoint.")
    parser.add_argument("--phase", type=str, default="val",
                        choices=["val", "test", "train", "trainval", "sample"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--limit", type=int,
                        help="Stop after this many samples")
    parser.add_argument("--occupancy_threshold", type=float,
                        default=DEFAULT_OCCUPANCY_THRESHOLD,
                        help="Magnitude above which a voxel counts as occupied "
                             "(default %(default)s)")
    parser.add_argument("--json", type=str,
                        help="Write the full results, per sample included, to this path")
    parser.add_argument("--trust_checkpoint", action="store_true",
                        help="Unpickle with weights_only=False. Needed only for "
                             "pre-2026-09-07 checkpoints, which stored a dataclass.")
    return parser


def load_checkpoint(path: str, trust: bool) -> dict:
    """Read a checkpoint, refusing to unpickle arbitrary objects unless asked."""
    try:
        return torch.load(path, map_location="cpu", weights_only=not trust)
    except Exception as error:
        if trust:
            raise
        raise RuntimeError(
            f"Could not load '{path}' with weights_only=True. Pre-2026-09-07 "
            f"checkpoints pickled a config dataclass; re-run with --trust_checkpoint "
            f"if you trust this file. Original error: {error}"
        ) from error


def resolve_model_config(checkpoint: dict, model_config_path: str | None) -> dict:
    """Take the model config from the CLI when given, otherwise from the checkpoint."""
    stored = checkpoint.get("config", {})
    stored = stored.get("model") if isinstance(stored, dict) else None

    if model_config_path:
        config = load_json(model_config_path)
        if stored and stored.get("backbone") != config.get("backbone"):
            print(f"WARNING: checkpoint was trained with backbone "
                  f"'{stored.get('backbone')}' but '{model_config_path}' asks for "
                  f"'{config.get('backbone')}'.")
        return config

    if not stored:
        raise ValueError(
            "Checkpoint stores no model config. Pass --model_config explicitly."
        )
    return stored


def load_model(config: dict, checkpoint: dict, device: torch.device) -> nn.Module:
    """Build the backbone and load its weights, refusing a partial match."""
    model = BackboneRegistry.build(config, device=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint does not match the model.\n"
            f"  missing:    {sorted(missing)[:6]}{' ...' if len(missing) > 6 else ''}\n"
            f"  unexpected: {sorted(unexpected)[:6]}{' ...' if len(unexpected) > 6 else ''}\n"
            f"Check --model_config matches how this checkpoint was trained."
        )
    return model.eval()


def build_eval_loader(phase: str, batch_size: int, num_workers: int):
    """Every sample once, in a fixed order.

    make_data_loader shuffles and drops the last partial batch, which is right for
    training and wrong for scoring. DSECfull disables augmentation on val/test/sample
    on its own.
    """
    dataset = DSECfull(phase)
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )


def compare(pred: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    """Score pred against ref."""
    # pred, ref: (B, T, H, W)
    m = compute_metrics(pred, ref)
    return {"raps": m["raps"], "ssim": m["ssim"], "l1": F.l1_loss(pred, ref).item()}


def signal_stats(x: torch.Tensor, threshold: float) -> dict[str, float]:
    """Peak, mean magnitude, exact-zero fraction and occupancy of a voxel grid."""
    # x: (B, T, H, W)
    return {
        "peak": x.abs().max().item(),
        "mean_abs": x.abs().mean().item(),
        "zero_frac": (x == 0).float().mean().item(),
        "occupancy": (x.abs() > threshold).float().mean().item(),
    }


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader,
    device:    torch.device,
    limit:     int | None = None,
    threshold: float = DEFAULT_OCCUPANCY_THRESHOLD,
) -> dict:
    """Score input and output against ground truth over the whole loader."""
    samples: list[dict] = []
    seen = 0
    progress = tqdm(loader, ncols=80, desc="Evaluating")
    for voxel, voxel_gt, _ in progress:
        source = voxel.to(device, non_blocking=True).float()
        target = voxel_gt.to(device, non_blocking=True).float()
        output = model(source)

        correction = output - source
        samples.append({
            "input_vs_gt":  compare(source, target),
            "output_vs_gt": compare(output, target),
            "output_vs_input": {
                "l1": F.l1_loss(output, source).item(),
                "relative_norm": (correction.norm() / source.norm().clamp_min(EPS)).item(),
            },
            "stats": {
                "input":  signal_stats(source, threshold),
                "output": signal_stats(output, threshold),
                "gt":     signal_stats(target, threshold),
            },
        })
        seen += source.shape[0]
        if limit is not None and seen >= limit:
            break

    return {"num_samples": len(samples), "samples": samples,
            "occupancy_threshold": threshold,
            "aggregate": aggregate(samples)}


def aggregate(samples: list[dict]) -> dict:
    """Average every field and count how often the output beats the input."""
    if not samples:
        raise ValueError("No samples evaluated.")
    n = len(samples)

    def mean(section: str, key: str) -> float:
        return sum(s[section][key] for s in samples) / n

    wins = {}
    for metric in METRIC_NAMES:
        better = sum(
            (s["output_vs_gt"][metric] < s["input_vs_gt"][metric])
            if LOWER_IS_BETTER[metric]
            else (s["output_vs_gt"][metric] > s["input_vs_gt"][metric])
            for s in samples
        )
        wins[metric] = better

    return {
        "input_vs_gt":  {k: mean("input_vs_gt", k) for k in METRIC_NAMES},
        "output_vs_gt": {k: mean("output_vs_gt", k) for k in METRIC_NAMES},
        "output_vs_input": {
            k: mean("output_vs_input", k) for k in ("l1", "relative_norm")
        },
        "stats": {
            source: {k: sum(s["stats"][source][k] for s in samples) / n
                     for k in STAT_NAMES}
            for source in ("input", "output", "gt")
        },
        "output_better_than_input": wins,
    }


def format_report(results: dict, header: dict) -> str:
    """Render the aggregate as a table."""
    agg = results["aggregate"]
    n = results["num_samples"]
    lines = [""]
    for key, value in header.items():
        lines.append(f"{key:<12} {value}")
    lines.append("")

    lines.append("Against ground truth")
    lines.append(f"  {'':<10}{'RAPS':>10}{'SSIM':>10}{'L1':>10}")
    for label, section in (("input", "input_vs_gt"), ("output", "output_vs_gt")):
        row = agg[section]
        lines.append(f"  {label:<10}" + "".join(f"{row[k]:>10.4f}" for k in METRIC_NAMES))

    delta = {k: agg["output_vs_gt"][k] - agg["input_vs_gt"][k] for k in METRIC_NAMES}
    lines.append(f"  {'change':<10}" + "".join(f"{delta[k]:>+10.4f}" for k in METRIC_NAMES))
    wins = agg["output_better_than_input"]
    counts = [f"{wins[k]}/{n}" for k in METRIC_NAMES]
    lines.append(f"  {'improved':<10}" + "".join(f"{c:>10}" for c in counts))
    lines.append("  (RAPS and L1 lower is better, SSIM higher is better)")

    lines.append("")
    lines.append("Output against input")
    ovi = agg["output_vs_input"]
    lines.append(f"  {'L1':<28}{ovi['l1']:>10.4f}")
    lines.append(f"  {'||output-input||/||input||':<28}{ovi['relative_norm']:>10.4f}")

    lines.append("")
    threshold = results["occupancy_threshold"]
    lines.append("Signal statistics")
    lines.append(f"  {'':<14}{'peak':>10}{'mean|x|':>10}{'exact 0':>10}"
                 f"{f'>{threshold:g}':>10}")
    for label in ("input", "output", "gt"):
        row = agg["stats"][label]
        lines.append(f"  {label:<14}{row['peak']:>10.3f}{row['mean_abs']:>10.4f}"
                     f"{row['zero_frac']:>10.3f}{row['occupancy']:>10.3f}")
    lines.append(f"  'exact 0' needs no threshold. Occupancy is the fraction above "
                 f"{threshold:g};")
    lines.append("  set --occupancy_threshold to probe a different magnitude.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)

    checkpoint = load_checkpoint(args.checkpoint, args.trust_checkpoint)
    model_config = resolve_model_config(checkpoint, args.model_config)
    model = load_model(model_config, checkpoint, device)
    loader = build_eval_loader(args.phase, args.batch_size, args.num_workers)

    results = evaluate(model, loader, device, args.limit, args.occupancy_threshold)

    header = {
        "checkpoint": args.checkpoint,
        "step": checkpoint.get("step", "unknown"),
        "backbone": model_config.get("backbone", "unknown"),
        "phase": args.phase,
        "samples": results["num_samples"],
        "device": str(device),
    }
    print(format_report(results, header))

    if args.json:
        payload = {"header": header, **results}
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
