import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from clip_features import CoVerCLIPFeatureExtractor
from config import DEFAULT_CONFIG
from main_calibration import read_jsonl, resolve_submission_dir, write_jsonl
from metrics import compute_metrics_from_rankings, write_metrics_json, write_online_submission
from scoring import pairwise_constraint_verification


def clean_pair(item: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    desired = str(item.get("desired", "")).strip()
    confusing = str(item.get("confusing", "")).strip()
    if desired and confusing:
        return desired, confusing
    return None


def get_pairs(row: Dict[str, Any]) -> List[Tuple[str, str]]:
    decomposition = row.get("decomposition") or {}
    pairs = []
    for item in decomposition.get("contrastive_pairs", []) or []:
        if isinstance(item, dict):
            pair = clean_pair(item)
            if pair is not None:
                pairs.append(pair)
    return pairs


def collect_unique_candidate_images(rows: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    image_map = {}
    for row in rows:
        calibration = row.get("calibration") or {}
        names = calibration.get("retrieved_image_names") or []
        paths = calibration.get("retrieved_image_paths") or []
        for name, path in zip(names, paths):
            if name not in image_map:
                image_map[str(name)] = path
    return image_map


def collect_unique_pair_texts(rows: Sequence[Dict[str, Any]]) -> List[str]:
    texts = []
    seen = set()
    for row in rows:
        for desired, confusing in get_pairs(row):
            for text in (desired, confusing):
                key = text.lower()
                if key not in seen:
                    texts.append(text)
                    seen.add(key)
    return texts


def load_gallery_feature_cache(path: Path) -> Dict[str, torch.Tensor]:
    cache = torch.load(path, map_location="cpu")
    features = cache["features"]
    names = [str(name) for name in cache["image_names"]]
    return {name: features[index] for index, name in enumerate(names)}


def encode_candidate_images(
    extractor: CoVerCLIPFeatureExtractor,
    rows: Sequence[Dict[str, Any]],
    batch_size: int,
    gallery_cache: Optional[Path],
) -> Dict[str, torch.Tensor]:
    cached_features = {}
    if gallery_cache is not None and gallery_cache.exists():
        cached_features = load_gallery_feature_cache(gallery_cache)
    image_map = collect_unique_candidate_images(rows)
    missing = [(name, path) for name, path in image_map.items() if name not in cached_features]
    if missing:
        names = [item[0] for item in missing]
        paths = [item[1] for item in missing]
        features = extractor.extract_image_features(paths, batch_size=batch_size).detach().cpu()
        for name, feature in zip(names, features):
            cached_features[name] = feature
    return cached_features


def encode_pair_texts(
    extractor: CoVerCLIPFeatureExtractor,
    rows: Sequence[Dict[str, Any]],
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    texts = collect_unique_pair_texts(rows)
    if not texts:
        return {}
    features = extractor.extract_text_features(texts, batch_size=batch_size).detach().cpu()
    return {text.lower(): feature for text, feature in zip(texts, features)}


def make_row_tensors(
    row: Dict[str, Any],
    image_feature_map: Dict[str, torch.Tensor],
    text_feature_map: Dict[str, torch.Tensor],
    verification_topk: Optional[int],
) -> Optional[Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    calibration = row.get("calibration") or {}
    names = [str(name) for name in (calibration.get("retrieved_image_names") or [])]
    scores = calibration.get("retrieved_scores") or []
    if verification_topk is not None:
        names = names[:verification_topk]
        scores = scores[:verification_topk]
    pairs = get_pairs(row)
    if not names or not scores or not pairs:
        return None

    image_features = []
    kept_names = []
    kept_scores = []
    for name, score in zip(names, scores):
        feature = image_feature_map.get(name)
        if feature is not None:
            image_features.append(feature)
            kept_names.append(name)
            kept_scores.append(float(score))
    if not image_features:
        return None

    desired_features = []
    confusing_features = []
    for desired, confusing in pairs:
        desired_feature = text_feature_map.get(desired.lower())
        confusing_feature = text_feature_map.get(confusing.lower())
        if desired_feature is not None and confusing_feature is not None:
            desired_features.append(desired_feature)
            confusing_features.append(confusing_feature)
    if not desired_features:
        return None

    return (
        kept_names,
        torch.stack(image_features, dim=0),
        torch.tensor(kept_scores, dtype=torch.float32),
        torch.stack(desired_features, dim=0),
        torch.stack(confusing_features, dim=0),
    )


def verify_one_row(
    row: Dict[str, Any],
    image_feature_map: Dict[str, torch.Tensor],
    text_feature_map: Dict[str, torch.Tensor],
    rho: float,
    verification_topk: Optional[int],
) -> Dict[str, Any]:
    tensors = make_row_tensors(row, image_feature_map, text_feature_map, verification_topk)
    calibration = row.get("calibration") or {}
    original_names = calibration.get("retrieved_image_names") or []
    original_scores = calibration.get("retrieved_scores") or []
    if tensors is None:
        row["verification"] = {
            "verified": False,
            "reason": "missing contrastive pairs or candidate/text features",
            "rho": rho,
            "retrieved_image_names": original_names,
            "retrieved_scores": original_scores,
        }
        row["retrieved_image_names"] = original_names
        row["retrieved_scores"] = original_scores
        return row

    names, candidate_features, calibrated_scores, desired_features, confusing_features = tensors
    output = pairwise_constraint_verification(
        candidate_image_features=candidate_features,
        calibrated_topk_scores=calibrated_scores,
        desired_text_features=desired_features,
        confusing_text_features=confusing_features,
        rho=rho,
    )
    final_scores = output.final_scores.squeeze(0).detach().cpu()
    order = torch.argsort(final_scores, descending=True)
    reranked_names = [names[index] for index in order.tolist()]
    reranked_scores = final_scores[order].tolist()
    verification_scores = output.verification_scores.squeeze(0).detach().cpu()

    row["verification"] = {
        "verified": True,
        "rho": rho,
        "verification_topk": len(names),
        "num_pairs": desired_features.shape[0],
        "retrieved_image_names": reranked_names,
        "retrieved_scores": reranked_scores,
        "verification_scores_by_calibration_order": verification_scores.tolist(),
        "final_scores_by_calibration_order": final_scores.tolist(),
    }
    row["retrieved_image_names"] = reranked_names
    row["retrieved_scores"] = reranked_scores
    return row


def run(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.calibration))
    if args.limit > 0:
        rows = rows[args.start : args.start + args.limit]
    elif args.start > 0:
        rows = rows[args.start :]
    extractor = CoVerCLIPFeatureExtractor(
        model_name=args.clip_model,
        library=args.clip_library,
        device=args.device,
        local_model_dir=args.local_model_dir,
        preprocess_type=args.preprocess_type,
        targetpad_ratio=args.targetpad_ratio,
    )
    gallery_cache = Path(args.gallery_cache) if args.gallery_cache else None
    image_features = encode_candidate_images(extractor, rows, args.batch_size, gallery_cache)
    text_features = encode_pair_texts(extractor, rows, args.text_batch_size)

    calibration_rows = copy.deepcopy(rows)
    output_rows = []
    for index, row in enumerate(rows):
        output_rows.append(
            verify_one_row(
                row=row,
                image_feature_map=image_features,
                text_feature_map=text_features,
                rho=args.rho,
                verification_topk=args.verification_topk,
            )
        )
        if (index + 1) % args.log_every == 0:
            print(f"verified {index + 1}/{len(rows)}")

    write_jsonl(Path(args.output), output_rows)
    calibration_metrics = compute_metrics_from_rankings(calibration_rows, args.dataset)
    verification_metrics = compute_metrics_from_rankings(output_rows, args.dataset)
    if calibration_metrics or verification_metrics:
        metrics_path = Path(args.output).with_suffix(".metrics.json")
        payload = {
            "dataset": args.dataset,
            "num_queries": len(output_rows),
            "params": {
                "rho": args.rho,
                "verification_topk": args.verification_topk,
                "clip_model": args.clip_model,
                "clip_library": args.clip_library,
            },
            "metrics": {
                "calibration": calibration_metrics,
                "verification": verification_metrics,
            },
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
    parser = argparse.ArgumentParser("CoVer-CIR pairwise verification")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--dataset", required=True, choices=["cirr", "circo", "fashioniq", "fashioniq_dress", "fashioniq_shirt", "fashioniq_toptee"])
    parser.add_argument("--split", default=cfg.dataset.default_split)
    parser.add_argument("--output", required=True)
    parser.add_argument("--submission-dir", default=None)
    parser.add_argument("--gallery-cache", default=None)
    parser.add_argument("--clip-model", default=cfg.clip.model_name)
    parser.add_argument("--clip-library", default=cfg.clip.library, choices=["openai", "open_clip"])
    parser.add_argument("--local-model-dir", default=cfg.clip.local_model_dir)
    parser.add_argument("--preprocess-type", default=cfg.clip.preprocess_type, choices=["clip", "targetpad"])
    parser.add_argument("--targetpad-ratio", type=float, default=cfg.clip.targetpad_ratio)
    parser.add_argument("--device", default=None)
    parser.add_argument("--rho", type=float, default=cfg.verification.rho)
    parser.add_argument("--verification-topk", type=int, default=cfg.verification.topk)
    parser.add_argument("--batch-size", type=int, default=cfg.verification.image_batch_size)
    parser.add_argument("--text-batch-size", type=int, default=cfg.verification.text_batch_size)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--log-every", type=int, default=cfg.verification.log_every)
    return parser


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
