import os

from evals.metrics.mia.all_attacks import AllAttacks
from evals.metrics.mia.loss import LOSSAttack
from evals.metrics.mia.reference import ReferenceAttack
from evals.metrics.mia.zlib import ZLIBAttack
from evals.metrics.mia.min_k import MinKProbAttack
from evals.metrics.mia.min_k_plus_plus import MinKPlusPlusAttack
from evals.metrics.mia.gradnorm import GradNormAttack

from sklearn.metrics import roc_auc_score


import numpy as np


def get_attacker(attack: str):
    mapping = {
        AllAttacks.LOSS: LOSSAttack,
        AllAttacks.REFERENCE_BASED: ReferenceAttack,
        AllAttacks.ZLIB: ZLIBAttack,
        AllAttacks.MIN_K: MinKProbAttack,
        AllAttacks.MIN_K_PLUS_PLUS: MinKPlusPlusAttack,
        AllAttacks.GRADNORM: GradNormAttack,
    }
    attack_cls = mapping.get(attack, None)
    if attack_cls is None:
        raise ValueError(f"Attack {attack} not found")
    return attack_cls


def mia_auc(attack_cls, model, data, collator, batch_size, **kwargs):
    """
    Compute the MIA attack metrics for all available datasets.

    Parameters:
      - attack_cls: the attack class to use.
      - model: the target model.
      - data: a dict with keys for datasets (e.g., "forget", "holdout", "retain").
      - collator: data collator.
      - batch_size: batch size.
      - kwargs: additional optional parameters (e.g. k, p, tokenizer, reference_model).

    Returns a dict containing the attack outputs for all available datasets.

    Note: This computes metrics (e.g., gradient norms) for each dataset separately.
    The top-level agg_value stays numeric for downstream metrics that expect a
    summary value, while the per-dataset values remain available under each key.
    """
    checkpoint_dir = kwargs.pop("checkpoint_dir", None)
    output_dir = kwargs.pop("output_dir", None)

    # Ensure checkpoint_dir is a usable path string if provided
    if checkpoint_dir is not None:
        if not isinstance(checkpoint_dir, (str, bytes, os.PathLike)):
            if hasattr(checkpoint_dir, "path"):
                checkpoint_dir = str(checkpoint_dir.path)
            elif isinstance(checkpoint_dir, dict) and "path" in checkpoint_dir:
                checkpoint_dir = str(checkpoint_dir["path"])
            else:
                # If it's a non-matching dict/object, clear it to use output_dir fallback
                checkpoint_dir = None

    if checkpoint_dir is None and output_dir is not None:
        checkpoint_dir = os.path.join(str(output_dir), "checkpoints", attack_cls.__name__.replace("Attack", "").lower())
    
    # Build attack arguments from common parameters and any extras.
    attack_args = {
        "model": model,
        "collator": collator,
        "batch_size": batch_size,
    }
    attack_args.update(kwargs)

    # Calculate attack metrics for all available datasets
    output = {}
    
    # Unified Pool Fix: If 'data' is a single dataset object (not a mapping), 
    # wrap it in a dict to avoid iterating over samples as keys.
    if not isinstance(data, (dict, list)) and hasattr(data, "__getitem__"):
        data = {"baseline_pool": data}
        
    for key in data:
        subset_attack_args = dict(attack_args)
        if checkpoint_dir is not None:
            subset_attack_args["checkpoint_dir"] = os.path.join(str(checkpoint_dir), str(key))
        output[key] = attack_cls(data=data[key], **subset_attack_args).attack()
    
    # Calculate ROC-AUC between subsets if possible.
    # Logic: Member = 0, Non-Member = 1 (since higher scores = less likely member)
    def compute_auc(member_scores, non_member_scores):
        if not member_scores or not non_member_scores:
            return None
        y_true = [0] * len(member_scores) + [1] * len(non_member_scores)
        y_score = member_scores + non_member_scores
        try:
            return float(roc_auc_score(y_true, y_score))
        except Exception:
            return None

    scores = {}
    for key, val in output.items():
        if isinstance(val, dict) and "value_by_index" in val:
            scores[key] = [v["score"] for v in val["value_by_index"].values()]

    if "forget" in scores:
        if "holdout" in scores:
            output["auc"] = compute_auc(scores["forget"], scores["holdout"])
        if "retain" in scores:
            output["auc_retain"] = compute_auc(scores["forget"], scores["retain"])

    # Preserve a numeric aggregate for downstream metrics such as privleak.
    # Now defaults to ROC-AUC if available, otherwise forget mean.
    if "auc" in output and output["auc"] is not None:
        output["agg_value"] = output["auc"]
    elif "auc_retain" in output and output["auc_retain"] is not None:
        output["agg_value"] = output["auc_retain"]
    elif "forget" in output and output["forget"].get("agg_value") is not None:
        output["agg_value"] = output["forget"]["agg_value"]
    else:
        output["agg_value"] = next(
            (
                subset_output["agg_value"]
                for subset_output in output.values()
                if isinstance(subset_output, dict)
                and subset_output.get("agg_value") is not None
            ),
            None,
        )
    return output
