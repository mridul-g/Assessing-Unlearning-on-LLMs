import logging
import json
import os
import hashlib
from collections import defaultdict
from itertools import islice
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from evals.metrics.utils import (
    aggregate_to_1D,
    evaluate_probability,
    eval_text_similarity,
    run_batchwise_evals,
    to_host,
    tokenwise_vocab_logprobs,
)
from evals.metrics.base import unlearning_metric

# Supress the info messages logged while calculating rouge using rouge_scorer
logging.getLogger("absl").setLevel(logging.WARNING)
logger = logging.getLogger("evaluator")


def _probability_cache_file(data, model, batch_size, output_dir):
    if output_dir is None:
        return None

    dataset_name = type(data).__name__
    question_key = getattr(data, "question_key", None)
    answer_key = getattr(data, "answer_key", None)
    predict_with_generate = getattr(data, "predict_with_generate", None)
    data_len = len(data)

    hf_data = getattr(data, "data", None)
    data_fingerprint = getattr(hf_data, "_fingerprint", None)

    model_name = getattr(getattr(model, "config", None), "_name_or_path", None)

    cache_identity = {
        "dataset_name": dataset_name,
        "question_key": question_key,
        "answer_key": answer_key,
        "predict_with_generate": predict_with_generate,
        "data_len": data_len,
        "data_fingerprint": data_fingerprint,
        "batch_size": int(batch_size),
        "model_name": model_name,
    }
    key = hashlib.sha1(
        json.dumps(cache_identity, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]

    cache_dir = os.path.join(output_dir, "cache", "probability")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{key}.json")


@unlearning_metric(name="probability")
def probability(model, **kwargs):
    """Compute the probabilities by data points and report aggregated average"""
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]
    output_dir = kwargs.get("output_dir", None)

    cache_file = _probability_cache_file(
        data=data,
        model=model,
        batch_size=batch_size,
        output_dir=output_dir,
    )
    if cache_file is not None and os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cached = json.load(f)
            if isinstance(cached, dict) and "value_by_index" in cached and "agg_value" in cached:
                logger.info("[prob-cache] loaded cached probability values from %s", cache_file)
                print(f"[prob-cache] loaded cached probability values from {cache_file}")
                return {
                    "agg_value": float(cached["agg_value"]),
                    "value_by_index": {
                        int(idx): val for idx, val in cached["value_by_index"].items()
                    },
                }
        except Exception:
            logger.warning("Failed to load probability cache from %s; recomputing.", cache_file)

    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)

    fun_args = {}
    scores_by_index = run_batchwise_evals(
        model, dataloader, evaluate_probability, fun_args, "Calculating loss"
    )
    # Ensure nested tensor outputs (including CUDA tensors) are host-native.
    scores_by_index = {
        idx: (to_host(evals) if evals is not None else None)
        for idx, evals in scores_by_index.items()
    }

    prob_values = np.array(
        [
            evals["prob"]
            for evals in scores_by_index.values()
            if evals["prob"] is not None
        ],
        dtype=np.float32,
    )
    prob_values = aggregate_to_1D(prob_values)
    result = {"agg_value": float(np.mean(prob_values)), "value_by_index": scores_by_index}

    if cache_file is not None:
        try:
            with open(cache_file, "w") as f:
                json.dump(
                    {
                        "agg_value": result["agg_value"],
                        "value_by_index": {str(k): v for k, v in result["value_by_index"].items()},
                    },
                    f,
                    indent=2,
                )
            logger.info("[prob-cache] saved probability values to %s", cache_file)
            print(f"[prob-cache] saved probability values to {cache_file}")
        except Exception:
            logger.warning("Failed to write probability cache to %s.", cache_file)

    return result


@unlearning_metric(name="probability_w_options")
def probability_w_options(model, **kwargs):
    """Normalize probabilities of correct answers against false answers for
    open-ended datasets, returning the aggregated value and per-index probabilities."""
    correct_answer_results = kwargs["pre_compute"]["correct"]["value_by_index"]
    wrong_answer_results = kwargs["pre_compute"]["wrong"]["value_by_index"]

    correct_indices = list(correct_answer_results.keys())
    wrong_indices = list(wrong_answer_results.keys())
    assert correct_indices == wrong_indices

    # Filter out None values from both correct and wrong answers
    filtered_indices = [
        idx
        for idx in correct_indices
        if correct_answer_results[idx] is not None
        and wrong_answer_results[idx] is not None
    ]
    correct = np.array(
        [to_host(correct_answer_results[idx]["prob"]) for idx in filtered_indices]
    )
    all_wrong = np.array(
        [to_host(wrong_answer_results[idx]["prob"]) for idx in filtered_indices]
    )
    wrong = np.sum(all_wrong, axis=tuple(range(1, all_wrong.ndim)))
    probs = correct / (correct + wrong + 1e-10)

    value_by_index = dict(zip(correct_indices, [{"prob": val} for val in probs]))
    return {"agg_value": np.mean(probs), "value_by_index": value_by_index}


@unlearning_metric(name="rouge")
def rouge(model, **kwargs):
    """Calculate ROUGE metrics and return the aggregated value along with per-index scores."""
    tokenizer = kwargs["tokenizer"]
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]
    generation_args = kwargs["generation_args"]
    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)

    fun_args = {"tokenizer": tokenizer, "generation_args": generation_args}
    scores_by_index = run_batchwise_evals(
        model,
        dataloader,
        eval_text_similarity,
        fun_args,
        "Calculating text similarity",
    )
    rouge_values = np.array(
        [
            evals[kwargs["rouge_type"]]
            for evals in scores_by_index.values()
            if evals[kwargs["rouge_type"]] is not None
        ]
    )
    rouge_values = aggregate_to_1D(rouge_values)
    return {
        "agg_value": np.mean(rouge_values),
        "value_by_index": scores_by_index,
    }


@unlearning_metric(name="truth_ratio")
def truth_ratio(model, **kwargs):
    """Compute the truth ratio, aggregating false/true scores, and
    return the aggregated value."""

    # Forget data: It is better if false and true are equally likely,
    # i.e., tr=false/true is closest to 1.
    def closer_to_1_better(arr):
        return np.mean(np.minimum(arr, 1 / (arr + 1e-10)))

    # Non-forget data: It is better if tr=false/true is lower, i.e.,
    # 1-tr is higher.
    def true_better(arr):
        return np.mean(np.maximum(0, 1 - arr))

    # Extent of knowledge (as used in OpenUnlearning paper's meta-evaluation) uses tr=true/(true+false)
    def prob_mean(arr):
        return np.mean(arr)

    if kwargs["aggregator"] == "closer_to_1_better":
        aggregator = closer_to_1_better
    elif kwargs["aggregator"] == "true_better":
        aggregator = true_better
    elif kwargs["aggregator"] == "prob_mean":
        aggregator = prob_mean
    else:
        raise ValueError(f"Invalid truth ratio aggregator: {kwargs['aggregator']}")

    correct_answer_results = kwargs["pre_compute"]["correct"]["value_by_index"]
    wrong_answer_results = kwargs["pre_compute"]["wrong"]["value_by_index"]

    correct_indices = set(correct_answer_results.keys())
    wrong_indices = set(wrong_answer_results.keys())
    
    # Filter out None values and use intersection of indices for robustness (especially during smoke runs)
    filtered_indices = sorted([
        idx for idx in correct_indices.intersection(wrong_indices)
        if correct_answer_results[idx] is not None
        and wrong_answer_results[idx] is not None
    ])
    correct_avg_losses = [
        to_host(correct_answer_results[idx]["avg_loss"]) for idx in filtered_indices
    ]
    wrong_avg_losses = [
        to_host(wrong_answer_results[idx]["avg_loss"]) for idx in filtered_indices
    ]

    correct_avg_losses = aggregate_to_1D(np.array(correct_avg_losses))
    wrong_avg_losses = aggregate_to_1D(np.array(wrong_avg_losses))

    correct_prob = np.exp(-correct_avg_losses)
    wrong_prob = np.exp(-wrong_avg_losses)

    if kwargs["aggregator"] != "prob_mean":
        # Original definition from TOFU: wrong / correct
        truth_ratios = wrong_prob / (correct_prob + 1e-10)
    else:
        # New definition from OpenUnlearning: correct / (correct + wrong)
        truth_ratios = correct_prob / (correct_prob + wrong_prob + 1e-10)

    value_by_index = dict(
        zip(correct_indices, [{"score": val} for val in truth_ratios])
    )
    truth_ratio_stats = np.array([evals["score"] for evals in value_by_index.values()])
    forget_tr_avg = aggregator(truth_ratio_stats)
    return {"agg_value": forget_tr_avg, "value_by_index": value_by_index}


@unlearning_metric(name="exact_memorization")
def exact_memorization(model, **kwargs):
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]
    checkpoint_dir = kwargs.get("checkpoint_dir", None)
    checkpoint_every_n_batches = int(kwargs.get("checkpoint_every_n_batches", 100))
    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)

    def _exact_memorization(model, batch):
        log_probs_batch, labels_batch = tokenwise_vocab_logprobs(
            model, batch, grad=False, return_labels=True
        )
        em_batch = []
        for log_probs, labels in zip(log_probs_batch, labels_batch):
            valid_len = len(labels)
            if valid_len == 0:
                # Rarely, tokenization can result in a mismatch with no valid target
                # tokens for loss computation (see preprocess_chat_instance() for
                # reference). Since this condition makes no sense in terms of
                # computing EM, we just choose to set EM=None
                logger.warning(
                    "EM score for an instance is marked None, due to "
                    "tokenization issues that resulted in no valid target tokens."
                )
                em_batch.append({"score": None})
            else:
                preds = torch.argmax(log_probs, dim=-1)
                em_score = (preds == labels).sum() / valid_len
                em_batch.append({"score": em_score.item()})
        return em_batch

    checkpoint_file = None
    start_batch_idx = 0
    scores_by_index = {}
    fallback_indices = list(range(len(data)))
    if hasattr(data, "data") and hasattr(data.data, "column_names") and "index" in data.data.column_names:
        fallback_indices = [int(idx) for idx in data.data["index"]]

    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_dir, "exact_memorization_checkpoint.json")
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, "r") as f:
                checkpoint = json.load(f)

            if checkpoint.get("done", False):
                return {
                    "agg_value": checkpoint.get("agg_value", None),
                    "value_by_index": {
                        int(k): v for k, v in checkpoint.get("value_by_index", {}).items()
                    },
                }

            start_batch_idx = int(checkpoint.get("next_batch_idx", 0))
            scores_by_index = {
                int(k): v for k, v in checkpoint.get("value_by_index", {}).items()
            }

    fallback_cursor = start_batch_idx * batch_size

    for batch_idx, batch in enumerate(
        tqdm(
            islice(dataloader, start_batch_idx, None),
            desc="Calculating EM",
            total=max(len(dataloader) - start_batch_idx, 0),
        ),
        start=start_batch_idx,
    ):
        if "input_ids" in batch:
            mini_batches = {"0": batch}
        else:
            mini_batches = batch

        first_mini_batch = next(iter(mini_batches.values()))
        current_batch_size = len(first_mini_batch["input_ids"])
        guessed_indices = fallback_indices[
            fallback_cursor : fallback_cursor + current_batch_size
        ]
        fallback_cursor += current_batch_size

        per_answer_scores = {}
        answer_order = []
        for intra_item_idx, mini_batch in mini_batches.items():
            answer_order.append(intra_item_idx)
            mini_batch = dict(mini_batch)
            if "index" in mini_batch:
                data_indices = mini_batch.pop("index").tolist()
            else:
                data_indices = guessed_indices
            batch_evals = _exact_memorization(model=model, batch=mini_batch)
            per_answer_scores[intra_item_idx] = dict(zip(data_indices, batch_evals))

        if len(per_answer_scores) == 1:
            indexwise_batch_evals = next(iter(per_answer_scores.values()))
        else:
            indexwise_batch_evals = {}
            all_indices = set()
            for answer_scores in per_answer_scores.values():
                all_indices.update(answer_scores.keys())
            for idx in all_indices:
                score_list = []
                for intra_item_idx in answer_order:
                    score = per_answer_scores[intra_item_idx].get(idx, {}).get("score", None)
                    score_list.append(score)
                indexwise_batch_evals[idx] = {"score": score_list}

        scores_by_index |= indexwise_batch_evals

        if checkpoint_file is not None and (
            (batch_idx + 1) % checkpoint_every_n_batches == 0
            or batch_idx + 1 == len(dataloader)
        ):
            em_values_now = [
                evals["score"] for evals in scores_by_index.values() if evals["score"] is not None
            ]
            agg_value_now = (
                float(np.mean(aggregate_to_1D(np.array(em_values_now))))
                if em_values_now
                else None
            )
            with open(checkpoint_file, "w") as f:
                json.dump(
                    {
                        "done": False,
                        "next_batch_idx": batch_idx + 1,
                        "value_by_index": {str(k): v for k, v in scores_by_index.items()},
                        "agg_value": agg_value_now,
                    },
                    f,
                    indent=2,
                )

    em_values = np.array(
        [
            evals["score"]
            for evals in scores_by_index.values()
            if evals["score"] is not None
        ]
    )
    em_values = aggregate_to_1D(em_values)
    agg_value = np.mean(em_values)

    if checkpoint_file is not None:
        with open(checkpoint_file, "w") as f:
            json.dump(
                {
                    "done": True,
                    "next_batch_idx": len(dataloader),
                    "value_by_index": {str(k): v for k, v in scores_by_index.items()},
                    "agg_value": float(agg_value),
                },
                f,
                indent=2,
            )

    return {"agg_value": agg_value, "value_by_index": scores_by_index}


@unlearning_metric(name="extraction_strength")
def extraction_strength(model, **kwargs):
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]
    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)

    def _extraction_strength(model, batch):
        log_probs_batch, labels_batch = tokenwise_vocab_logprobs(
            model, batch, grad=False, return_labels=True
        )
        es_batch = []
        for log_probs, labels in zip(log_probs_batch, labels_batch):
            valid_len = len(labels)
            preds = torch.argmax(log_probs, dim=-1)
            for k in range(valid_len):
                suff_preds = preds[k:]
                suff_labels = labels[k:]
                if torch.equal(suff_preds, suff_labels):
                    break
            if valid_len == 0:
                # Rarely, tokenization can result in a mismatch with no valid target
                # tokens for loss computation (see preprocess_chat_instance() for
                # reference). Since this condition makes no sense in terms of
                # computing ES, we just choose to set ES=None
                logger.warning(
                    "ES score for an instance is marked None, due to "
                    "tokenization issues that resulted in no valid target tokens."
                )
                es_batch.append({"score": 0})
            else:
                es_score = 1 - (k / valid_len)
                es_batch.append({"score": es_score})
        return es_batch

    fun_args = {}
    scores_by_index = run_batchwise_evals(
        model, dataloader, _extraction_strength, fun_args, "Calculating ES"
    )
    es_values = np.array(
        [
            evals["score"]
            for evals in scores_by_index.values()
            if evals["score"] is not None
        ]
    )
    es_values = aggregate_to_1D(es_values)
    return {"agg_value": np.mean(es_values), "value_by_index": scores_by_index}
