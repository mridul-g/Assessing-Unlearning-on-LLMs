"""
Attack implementations.
"""

from transformers import AutoModelForCausalLM

from evals.metrics.base import unlearning_metric
from evals.metrics.mia.loss import LOSSAttack
from evals.metrics.mia.min_k import MinKProbAttack
from evals.metrics.mia.min_k_plus_plus import MinKPlusPlusAttack
from evals.metrics.mia.gradnorm import GradNormAttack
from evals.metrics.mia.zlib import ZLIBAttack
from evals.metrics.mia.reference import ReferenceAttack

from evals.metrics.mia.utils import mia_auc
import logging

logger = logging.getLogger("metrics")

## NOTE: all MIA attack statistics are signed as required in order to show the
# same trends as loss (higher the score on an example, less likely the membership)


@unlearning_metric(name="mia_loss")
def mia_loss(model, **kwargs):
    attack_kwargs = dict(kwargs)
    data = attack_kwargs.pop("data")
    collators = attack_kwargs.pop("collators")
    batch_size = attack_kwargs.pop("batch_size")
    return mia_auc(
        LOSSAttack,
        model,
        data=data,
        collator=collators,
        batch_size=batch_size,
        **attack_kwargs,
    )


@unlearning_metric(name="mia_min_k")
def mia_min_k(model, **kwargs):
    attack_kwargs = dict(kwargs)
    data = attack_kwargs.pop("data")
    collators = attack_kwargs.pop("collators")
    batch_size = attack_kwargs.pop("batch_size")
    k = attack_kwargs.pop("k")
    return mia_auc(
        MinKProbAttack,
        model,
        data=data,
        collator=collators,
        batch_size=batch_size,
        k=k,
        **attack_kwargs,
    )


@unlearning_metric(name="mia_min_k_plus_plus")
def mia_min_k_plus_plus(model, **kwargs):
    attack_kwargs = dict(kwargs)
    data = attack_kwargs.pop("data")
    collators = attack_kwargs.pop("collators")
    batch_size = attack_kwargs.pop("batch_size")
    k = attack_kwargs.pop("k")
    return mia_auc(
        MinKPlusPlusAttack,
        model,
        data=data,
        collator=collators,
        batch_size=batch_size,
        k=k,
        **attack_kwargs,
    )


@unlearning_metric(name="mia_gradnorm")
def mia_gradnorm(model, **kwargs):
    attack_kwargs = dict(kwargs)
    data = attack_kwargs.pop("data")
    collators = attack_kwargs.pop("collators")
    batch_size = attack_kwargs.pop("batch_size")
    p = attack_kwargs.pop("p")
    return mia_auc(
        GradNormAttack,
        model,
        data=data,
        collator=collators,
        batch_size=batch_size,
        p=p,
        **attack_kwargs,
    )


@unlearning_metric(name="mia_zlib")
def mia_zlib(model, **kwargs):
    attack_kwargs = dict(kwargs)
    data = attack_kwargs.pop("data")
    collators = attack_kwargs.pop("collators")
    batch_size = attack_kwargs.pop("batch_size")
    tokenizer = attack_kwargs.pop("tokenizer", None)
    return mia_auc(
        ZLIBAttack,
        model,
        data=data,
        collator=collators,
        batch_size=batch_size,
        tokenizer=tokenizer,
        **attack_kwargs,
    )


@unlearning_metric(name="mia_reference")
def mia_reference(model, **kwargs):
    if "reference_model_path" not in kwargs:
        raise ValueError("Reference model must be provided in kwargs")
    logger.info(f"Loading reference model from {kwargs['reference_model_path']}")
    reference_model = AutoModelForCausalLM.from_pretrained(
        kwargs["reference_model_path"],
        torch_dtype=model.dtype,
        device_map={"": model.device},
    )
    attack_kwargs = dict(kwargs)
    attack_kwargs.pop("reference_model_path", None)
    data = attack_kwargs.pop("data")
    collators = attack_kwargs.pop("collators")
    batch_size = attack_kwargs.pop("batch_size")
    return mia_auc(
        ReferenceAttack,
        model,
        data=data,
        collator=collators,
        batch_size=batch_size,
        reference_model=reference_model,
        **attack_kwargs,
    )
