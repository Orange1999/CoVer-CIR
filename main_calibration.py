import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from clip_features import CoVerCLIPFeatureExtractor
from config import DEFAULT_CONFIG
from decomposition import read_json
from metrics import compute_metrics_from_rankings, write_metrics_json, write_online_submission
from scoring import adaptive_negative_calibration


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_decomposition_rows(path: Path, keep_failed: bool = False) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    if keep_failed:
        return rows
    return [row for row in rows if row.get("ok", True) and row.get("decomposition")]


def build_gallery_rows(dataset: str, dataset_path: Path, split: str, fashioniq_types: Sequence[str]) -> List[Dict[str, str]]:
    key = dataset.lower()
    if key == "cirr":
        split_name = "test1" if split == "test" else split
        name_to_relpath = read_json(dataset_path / "cirr" / "image_splits" / f"split.rc2.{split_name}.json")
        name_to_path = {name: str(dataset_path / relpath) for name, relpath in name_to_relpath.items()}
        return [{"image_name": name, "image_path": path} for name, path in sorted(name_to_path.items())]

    if key == "fashioniq" or key.startswith("fashioniq_"):
        types = fashioniq_types if key == "fashioniq" else [key.split("_", 1)[1]]
        names = set()
        for dress_type in types:
            split_names = read_json(dataset_path / "image_splits" / f"split.{dress_type}.{split}.json")
            names.update(split_names)
        return [
            {"image_name": name, "image_path": str(dataset_path / "images" / f"{name}.png")}
            for name in sorted(names)
        ]

    if key == "circo":
        image_dir = dataset_path / "COCO2017_unlabeled" / "unlabeled2017"
        gallery = []
        for path in sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))):
            gallery.append({"image_name": str(int(path.stem)), "image_path": str(path)})
        return gallery

    raise ValueError(f"Unsupported dataset: {dataset}")


def save_gallery_cache(cache_path: Path, features: torch.Tensor, gallery_rows: Sequence[Dict[str, str]], meta: Dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": features.cpu(),
            "image_names": [row["image_name"] for row in gallery_rows],
            "image_paths": [row["image_path"] for row in gallery_rows],
            "meta": meta,
        },
        cache_path,
    )


def load_gallery_cache(cache_path: Path) -> Dict[str, Any]:
    return torch.load(cache_path, map_location="cpu")


def encode_gallery(
    extractor: CoVerCLIPFeatureExtractor,
    gallery_rows: Sequence[Dict[str, str]],
    batch_size: int,
) -> torch.Tensor:
    image_paths = [row["image_path"] for row in gallery_rows]
    return extractor.extract_image_features(image_paths, batch_size=batch_size).cpu()


def make_negative_queries(
    extractor: CoVerCLIPFeatureExtractor,
    reference_image_path: str,
    negative_constraints: Sequence[str],
    beta: float,
) -> Optional[torch.Tensor]:
    constraints = [item.strip() for item in negative_constraints if item and item.strip()]
    if not constraints:
        return None
    queries = extractor.compose_image_text_features(reference_image_path, constraints, gamma=beta)
    return queries.unsqueeze(0)


def compute_calibrated_scores_for_query(
    extractor: CoVerCLIPFeatureExtractor,
    gallery_features: torch.Tensor,
    reference_image_path: str,
    positive_target: str,
    negative_constraints: Sequence[str],
    alpha: float,
    beta: float,
    lambda_weight: float,
    topk: int,
) -> Dict[str, Any]:
    positive_query = extractor.compose_image_text_features(reference_image_path, positive_target, gamma=alpha)
    negative_queries = make_negative_queries(extractor, reference_image_path, negative_constraints, beta)
    calibrated = adaptive_negative_calibration(
        gallery_image_features=gallery_features.to(extractor.device),
        positive_query_features=positive_query,
        negative_query_features=negative_queries,
        lambda_weight=lambda_weight,
        topk=topk,
    )
    return {
        "scores": calibrated.scores.squeeze(0).detach().cpu(),
        "positive_scores": calibrated.positive_scores.squeeze(0).detach().cpu(),
        "negative_weights": None if calibrated.negative_weights is None else calibrated.negative_weights.squeeze(0).detach().cpu(),
        "topk_indices": calibrated.topk_indices.squeeze(0).detach().cpu(),
        "topk_scores": calibrated.topk_scores.squeeze(0).detach().cpu(),
    }


def remove_reference_from_ranking(
    indices: torch.Tensor,
    scores: torch.Tensor,
    image_names: Sequence[str],
    reference_name: Optional[str],
    topk: int,
) -> tuple:
    if not reference_name:
        return indices[:topk], scores[:topk]
    kept_indices = []
    kept_scores = []
    for index, score in zip(indices.tolist(), scores.tolist()):
        if str(image_names[index]) != str(reference_name):
            kept_indices.append(index)
            kept_scores.append(score)
        if len(kept_indices) >= topk:
            break
    return torch.tensor(kept_indices, dtype=torch.long), torch.tensor(kept_scores, dtype=torch.float32)


def rank_from_scores(
    scores: torch.Tensor,
    image_names: Sequence[str],
    reference_name: Optional[str],
    topk: int,
    remove_reference: bool,
) -> tuple:
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    if remove_reference:
        return remove_reference_from_ranking(sorted_indices, sorted_scores, image_names, reference_name, topk)
    return sorted_indices[:topk], sorted_scores[:topk]


def resolve_submission_dir(output_path: Path, submission_dir: Optional[str]) -> Path:
    if submission_dir:
        return Path(submission_dir)
    return output_path.with_suffix("").parent / f"{output_path.stem}_submission"


def run(args: argparse.Namespace) -> None:
    decomposition_rows = load_decomposition_rows(Path(args.decomposition), keep_failed=args.keep_failed)
    if args.limit > 0:
        decomposition_rows = decomposition_rows[args.start : args.start + args.limit]
    elif args.start > 0:
        decomposition_rows = decomposition_rows[args.start :]

    extractor = CoVerCLIPFeatureExtractor(
        model_name=args.clip_model,
        library=args.clip_library,
        device=args.device,
        local_model_dir=args.local_model_dir,
        preprocess_type=args.preprocess_type,
        targetpad_ratio=args.targetpad_ratio,
    )

    gallery_cache = Path(args.gallery_cache) if args.gallery_cache else None
    if gallery_cache and gallery_cache.exists() and not args.overwrite_gallery_cache:
        cache = load_gallery_cache(gallery_cache)
        gallery_features = cache["features"]
        image_names = cache["image_names"]
        image_paths = cache["image_paths"]
    else:
        gallery_rows = build_gallery_rows(args.dataset, Path(args.dataset_path), args.split, args.fashioniq_types)
        gallery_features = encode_gallery(extractor, gallery_rows, args.batch_size)
        image_names = [row["image_name"] for row in gallery_rows]
        image_paths = [row["image_path"] for row in gallery_rows]
        if gallery_cache:
            save_gallery_cache(
                gallery_cache,
                gallery_features,
                gallery_rows,
                {
                    "dataset": args.dataset,
                    "dataset_path": args.dataset_path,
                    "split": args.split,
                    "clip_model": args.clip_model,
                    "clip_library": args.clip_library,
                    "preprocess_type": args.preprocess_type,
                },
            )

    output_rows = []
    for idx, row in enumerate(decomposition_rows):
        decomp = row.get("decomposition") or {}
        positive_target = decomp.get("positive_target")
        negative_constraints = decomp.get("negative_constraints") or []
        if not positive_target:
            continue
        scored = compute_calibrated_scores_for_query(
            extractor=extractor,
            gallery_features=gallery_features,
            reference_image_path=row["reference_image_path"],
            positive_target=positive_target,
            negative_constraints=negative_constraints,
            alpha=args.alpha,
            beta=args.beta,
            lambda_weight=args.lambda_weight,
            topk=min(args.internal_topk, len(image_names)),
        )
        rank_indices, rank_scores = rank_from_scores(
            scores=scored["scores"],
            image_names=image_names,
            reference_name=row.get("reference_name"),
            topk=args.topk,
            remove_reference=args.dataset.lower() == "cirr",
        )
        retrieved_names = [image_names[index] for index in rank_indices.tolist()]
        retrieved_paths = [image_paths[index] for index in rank_indices.tolist()]
        negative_weights = scored["negative_weights"]
        output_rows.append(
            {
                "sample_id": row.get("sample_id"),
                "dataset": row.get("dataset", args.dataset),
                "reference_name": row.get("reference_name"),
                "reference_image_path": row.get("reference_image_path"),
                "target_name": row.get("target_name"),
                "pair_id": row.get("pair_id"),
                "query_id": row.get("query_id"),
                "group_members": row.get("group_members"),
                "gt_img_ids": row.get("gt_img_ids"),
                "relative_caption": row.get("relative_caption"),
                "decomposition": decomp,
                "calibration": {
                    "alpha": args.alpha,
                    "beta": args.beta,
                    "lambda_weight": args.lambda_weight,
                    "positive_target": positive_target,
                    "negative_constraints": negative_constraints,
                    "negative_weights": None if negative_weights is None else negative_weights.tolist(),
                    "retrieved_image_names": retrieved_names,
                    "retrieved_image_paths": retrieved_paths,
                    "retrieved_scores": rank_scores.tolist(),
                },
                "retrieved_image_names": retrieved_names,
                "retrieved_scores": rank_scores.tolist(),
            }
        )
        if (idx + 1) % args.log_every == 0:
            print(f"processed {idx + 1}/{len(decomposition_rows)}")

    write_jsonl(Path(args.output), output_rows)
    metrics = compute_metrics_from_rankings(output_rows, args.dataset)
    if metrics:
        metrics_path = Path(args.output).with_suffix(".metrics.json")
        payload = {
            "dataset": args.dataset,
            "split": args.split,
            "num_queries": len(output_rows),
            "topk_saved": args.topk,
            "params": {
                "alpha": args.alpha,
                "beta": args.beta,
                "lambda_weight": args.lambda_weight,
                "clip_model": args.clip_model,
                "clip_library": args.clip_library,
            },
            "metrics": metrics,
        }
        write_metrics_json(metrics_path, payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    elif args.split == "test":
        submission_paths = write_online_submission(
            output_rows,
            args.dataset,
            resolve_submission_dir(Path(args.output), args.submission_dir),
        )
        if submission_paths:
            print(json.dumps({"submission": submission_paths}, indent=2, ensure_ascii=False))
    print(f"saved {len(output_rows)} rows to {args.output}")


def build_arg_parser() -> argparse.ArgumentParser:
    cfg = DEFAULT_CONFIG
    parser = argparse.ArgumentParser("CoVer-CIR calibrated retrieval")
    parser.add_argument("--decomposition", required=True)
    parser.add_argument("--dataset", required=True, choices=["cirr", "circo", "fashioniq", "fashioniq_dress", "fashioniq_shirt", "fashioniq_toptee"])
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--split", default=cfg.dataset.default_split)
    parser.add_argument("--output", required=True)
    parser.add_argument("--submission-dir", default=None)
    parser.add_argument("--gallery-cache", default=None)
    parser.add_argument("--overwrite-gallery-cache", action="store_true")
    parser.add_argument("--clip-model", default=cfg.clip.model_name)
    parser.add_argument("--clip-library", default=cfg.clip.library, choices=["openai", "open_clip"])
    parser.add_argument("--local-model-dir", default=cfg.clip.local_model_dir)
    parser.add_argument("--preprocess-type", default=cfg.clip.preprocess_type, choices=["clip", "targetpad"])
    parser.add_argument("--targetpad-ratio", type=float, default=cfg.clip.targetpad_ratio)
    parser.add_argument("--device", default=None)
    parser.add_argument("--fashioniq-types", nargs="+", default=list(cfg.dataset.fashioniq_types), choices=list(cfg.dataset.fashioniq_types))
    parser.add_argument("--alpha", type=float, default=cfg.calibration.alpha)
    parser.add_argument("--beta", type=float, default=cfg.calibration.beta)
    parser.add_argument("--lambda-weight", type=float, default=cfg.calibration.lambda_weight)
    parser.add_argument("--topk", type=int, default=cfg.calibration.saved_topk)
    parser.add_argument("--internal-topk", type=int, default=cfg.calibration.internal_topk)
    parser.add_argument("--batch-size", type=int, default=cfg.calibration.batch_size)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--keep-failed", action="store_true")
    parser.add_argument("--log-every", type=int, default=cfg.calibration.log_every)
    return parser


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
