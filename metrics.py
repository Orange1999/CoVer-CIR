import json
from pathlib import Path
from typing import Any, Dict, Sequence

from config import DEFAULT_CONFIG


def _names(row: Dict[str, Any], ranking_key: str = "retrieved_image_names"):
    return [str(name) for name in row.get(ranking_key, [])]


def _target(row: Dict[str, Any]):
    value = row.get("target_name")
    return None if value is None else str(value)


def compute_fashioniq_metrics_from_rankings(
    rows: Sequence[Dict[str, Any]],
    ranking_key: str = "retrieved_image_names",
) -> Dict[str, float]:
    cutoffs = DEFAULT_CONFIG.metrics.fashioniq_recall_cutoffs
    hits = {k: 0 for k in cutoffs}
    total = 0
    for row in rows:
        target = _target(row)
        if not target:
            continue
        total += 1
        names = _names(row, ranking_key)
        for k in cutoffs:
            hits[k] += int(target in names[:k])
    return {f"Recall@{k}": hits[k] / max(total, 1) * 100 for k in cutoffs}


def compute_cirr_metrics_from_rankings(
    rows: Sequence[Dict[str, Any]],
    ranking_key: str = "retrieved_image_names",
) -> Dict[str, float]:
    cutoffs = DEFAULT_CONFIG.metrics.cirr_recall_cutoffs
    group_cutoffs = DEFAULT_CONFIG.metrics.cirr_group_recall_cutoffs
    hits = {k: 0 for k in cutoffs}
    group_hits = {k: 0 for k in group_cutoffs}
    total = 0
    for row in rows:
        target = _target(row)
        if not target:
            continue
        total += 1
        names = _names(row, ranking_key)
        group = set(str(name) for name in row.get("group_members", []) if str(name))
        group_ranking = [name for name in names if name in group]
        for k in cutoffs:
            hits[k] += int(target in names[:k])
        for k in group_cutoffs:
            group_hits[k] += int(target in group_ranking[:k])
    metrics = {f"recall@{k}": hits[k] / max(total, 1) * 100 for k in cutoffs}
    metrics.update({f"group_recall@{k}": group_hits[k] / max(total, 1) * 100 for k in group_cutoffs})
    return metrics


def compute_circo_metrics_from_rankings(
    rows: Sequence[Dict[str, Any]],
    ranking_key: str = "retrieved_image_names",
) -> Dict[str, float]:
    cutoffs = DEFAULT_CONFIG.metrics.circo_cutoffs
    ap_sum = {k: 0.0 for k in cutoffs}
    recall_sum = {k: 0.0 for k in cutoffs}
    total = 0
    for row in rows:
        gt = [str(name) for name in row.get("gt_img_ids", []) if str(name)]
        if not gt:
            continue
        total += 1
        gt_set = set(gt)
        target = _target(row) or gt[0]
        names = _names(row, ranking_key)
        labels = [1 if name in gt_set else 0 for name in names[: DEFAULT_CONFIG.metrics.circo_ranking_depth]]
        for k in cutoffs:
            precision_sum = 0.0
            hit_count = 0
            for rank, label in enumerate(labels[:k], start=1):
                if label:
                    hit_count += 1
                    precision_sum += hit_count / rank
            ap_sum[k] += precision_sum / max(min(len(gt_set), k), 1)
            recall_sum[k] += int(target in names[:k])
    metrics = {f"mAP@{k}": ap_sum[k] / max(total, 1) * 100 for k in cutoffs}
    metrics.update({f"recall@{k}": recall_sum[k] / max(total, 1) * 100 for k in cutoffs})
    return metrics


def compute_metrics_from_rankings(
    rows: Sequence[Dict[str, Any]],
    dataset: str,
    ranking_key: str = "retrieved_image_names",
) -> Dict[str, float]:
    key = dataset.lower()
    if key == "cirr":
        return compute_cirr_metrics_from_rankings(rows, ranking_key)
    if key == "circo":
        return compute_circo_metrics_from_rankings(rows, ranking_key)
    if key == "fashioniq" or key.startswith("fashioniq_"):
        return compute_fashioniq_metrics_from_rankings(rows, ranking_key)
    raise ValueError(f"Unsupported dataset for metrics: {dataset}")


def write_online_submission(
    rows: Sequence[Dict[str, Any]],
    dataset: str,
    output_dir: Path,
    ranking_key: str = "retrieved_image_names",
) -> Dict[str, str]:
    key = dataset.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    if key == "cirr":
        return write_cirr_submission(rows, output_dir, ranking_key)
    if key == "circo":
        return write_circo_submission(rows, output_dir, ranking_key)
    return {}


def write_cirr_submission(
    rows: Sequence[Dict[str, Any]],
    output_dir: Path,
    ranking_key: str = "retrieved_image_names",
) -> Dict[str, str]:
    retrieval = {"version": "rc2", "metric": "recall"}
    subset = {"version": "rc2", "metric": "recall_subset"}
    for row in rows:
        pair_id = str(row.get("pair_id") or row.get("sample_id"))
        names = [name for name in _names(row, ranking_key) if name != str(row.get("reference_name"))]
        group = set(str(name) for name in row.get("group_members", []) if str(name))
        retrieval[pair_id] = names[:50]
        subset[pair_id] = [name for name in names if name in group][:3]

    retrieval_path = output_dir / "cirr_test_submission.json"
    subset_path = output_dir / "cirr_test_submission_subset.json"
    retrieval_path.write_text(json.dumps(retrieval, sort_keys=True), encoding="utf-8")
    subset_path.write_text(json.dumps(subset, sort_keys=True), encoding="utf-8")
    return {"cirr_recall": str(retrieval_path), "cirr_recall_subset": str(subset_path)}


def write_circo_submission(
    rows: Sequence[Dict[str, Any]],
    output_dir: Path,
    ranking_key: str = "retrieved_image_names",
) -> Dict[str, str]:
    submission = {}
    for row in rows:
        query_id = str(row.get("query_id") or row.get("sample_id"))
        submission[query_id] = _names(row, ranking_key)[:50]

    submission_path = output_dir / "circo_test_submission.json"
    submission_path.write_text(json.dumps(submission, sort_keys=True), encoding="utf-8")
    return {"circo_test": str(submission_path)}


def write_metrics_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
