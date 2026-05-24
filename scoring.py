from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from config import DEFAULT_CONFIG


def l2_normalize(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features.float(), dim=-1)


def zscore(values: torch.Tensor, dim: int = -1, eps: float = DEFAULT_CONFIG.numerical.eps) -> torch.Tensor:
    mean = values.mean(dim=dim, keepdim=True)
    std = values.std(dim=dim, keepdim=True, unbiased=False).clamp_min(eps)
    return (values - mean) / std


@dataclass
class CalibrationOutput:
    scores: torch.Tensor
    positive_scores: torch.Tensor
    negative_scores: Optional[torch.Tensor]
    negative_weights: Optional[torch.Tensor]
    topk_scores: Optional[torch.Tensor] = None
    topk_indices: Optional[torch.Tensor] = None


@dataclass
class VerificationOutput:
    final_scores: torch.Tensor
    verification_scores: torch.Tensor
    pair_margins: torch.Tensor


def adaptive_negative_calibration(
    gallery_image_features: torch.Tensor,
    positive_query_features: torch.Tensor,
    negative_query_features: Optional[torch.Tensor] = None,
    lambda_weight: float = DEFAULT_CONFIG.calibration.lambda_weight,
    topk: Optional[int] = None,
    eps: float = DEFAULT_CONFIG.numerical.eps,
) -> CalibrationOutput:
    gallery_image_features = l2_normalize(gallery_image_features)
    positive_query_features = l2_normalize(positive_query_features)
    positive_scores = positive_query_features @ gallery_image_features.T
    positive_scores_norm = zscore(positive_scores, dim=-1, eps=eps)

    if negative_query_features is None or negative_query_features.numel() == 0:
        calibrated_scores = positive_scores_norm
        output = CalibrationOutput(
            scores=calibrated_scores,
            positive_scores=positive_scores_norm,
            negative_scores=None,
            negative_weights=None,
        )
    else:
        negative_query_features = l2_normalize(negative_query_features)
        if negative_query_features.ndim == 2:
            negative_query_features = negative_query_features.unsqueeze(0)
        if positive_query_features.shape[0] == 1 and negative_query_features.shape[0] > 1:
            positive_scores_norm = positive_scores_norm.expand(negative_query_features.shape[0], -1)
        if negative_query_features.shape[0] == 1 and positive_query_features.shape[0] > 1:
            negative_query_features = negative_query_features.expand(positive_query_features.shape[0], -1, -1)

        negative_scores = torch.einsum("bmd,nd->bmn", negative_query_features, gallery_image_features)
        negative_scores_norm = zscore(negative_scores, dim=-1, eps=eps)
        positive_distribution = torch.softmax(positive_scores_norm, dim=-1)
        activations = (positive_distribution.unsqueeze(1) * torch.relu(negative_scores_norm)).sum(dim=-1)
        negative_weights = (activations + eps) / (activations.sum(dim=-1, keepdim=True) + eps * activations.shape[-1])
        penalty = (negative_weights.unsqueeze(-1) * torch.relu(negative_scores_norm)).sum(dim=1)
        calibrated_scores = positive_scores_norm - lambda_weight * penalty
        output = CalibrationOutput(
            scores=calibrated_scores,
            positive_scores=positive_scores_norm,
            negative_scores=negative_scores_norm,
            negative_weights=negative_weights,
        )

    if topk is not None:
        k = min(topk, output.scores.shape[-1])
        output.topk_scores, output.topk_indices = torch.topk(output.scores, k=k, dim=-1)
    return output


def gather_topk_features(gallery_image_features: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
    if topk_indices.ndim == 1:
        return gallery_image_features[topk_indices].unsqueeze(0)
    flat = gallery_image_features[topk_indices.reshape(-1)]
    return flat.reshape(*topk_indices.shape, gallery_image_features.shape[-1])


def pairwise_constraint_verification(
    candidate_image_features: torch.Tensor,
    calibrated_topk_scores: torch.Tensor,
    desired_text_features: torch.Tensor,
    confusing_text_features: torch.Tensor,
    rho: float = DEFAULT_CONFIG.verification.rho,
    eps: float = DEFAULT_CONFIG.numerical.eps,
) -> VerificationOutput:
    candidate_image_features = l2_normalize(candidate_image_features)
    desired_text_features = l2_normalize(desired_text_features)
    confusing_text_features = l2_normalize(confusing_text_features)

    if candidate_image_features.ndim == 2:
        candidate_image_features = candidate_image_features.unsqueeze(0)
    if desired_text_features.ndim == 2:
        desired_text_features = desired_text_features.unsqueeze(0)
    if confusing_text_features.ndim == 2:
        confusing_text_features = confusing_text_features.unsqueeze(0)
    if desired_text_features.shape[0] == 1 and candidate_image_features.shape[0] > 1:
        desired_text_features = desired_text_features.expand(candidate_image_features.shape[0], -1, -1)
    if confusing_text_features.shape[0] == 1 and candidate_image_features.shape[0] > 1:
        confusing_text_features = confusing_text_features.expand(candidate_image_features.shape[0], -1, -1)

    desired_scores = torch.einsum("bkd,brd->brk", candidate_image_features, desired_text_features)
    confusing_scores = torch.einsum("bkd,brd->brk", candidate_image_features, confusing_text_features)
    pair_margins = desired_scores - confusing_scores
    normalized_margins = zscore(pair_margins, dim=-1, eps=eps)
    verification_scores = normalized_margins.mean(dim=1)

    if calibrated_topk_scores.ndim == 1:
        calibrated_topk_scores = calibrated_topk_scores.unsqueeze(0)
    final_scores = zscore(calibrated_topk_scores, dim=-1, eps=eps) + rho * zscore(verification_scores, dim=-1, eps=eps)
    return VerificationOutput(
        final_scores=final_scores,
        verification_scores=verification_scores,
        pair_margins=pair_margins,
    )


def rerank_topk_indices(topk_indices: torch.Tensor, final_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    order = torch.argsort(final_scores, dim=-1, descending=True)
    reranked_scores = torch.gather(final_scores, dim=-1, index=order)
    reranked_indices = torch.gather(topk_indices, dim=-1, index=order)
    return reranked_indices, reranked_scores
