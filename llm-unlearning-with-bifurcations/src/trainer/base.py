# Modified from https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py

import logging
import os
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset
from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

logger = logging.getLogger(__name__)

# When using custom evaluators without an eval dataset, pass a dummy value
# to prevent Trainer from raising on eval_dataset=None when eval_strategy is set
_EVAL_PLACEHOLDER = "_EVAL_PLACEHOLDER"


class FinetuneTrainer(Trainer):
    def __init__(self, evaluators=None, template_args=None, *args, **kwargs):
        self.evaluators = evaluators
        self.template_args = template_args
        if kwargs.get("eval_dataset") is None and evaluators:
            kwargs["eval_dataset"] = _EVAL_PLACEHOLDER
        super().__init__(*args, **kwargs)

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        trial: Dict[str, Any] = None,
    ) -> Dict[str, float]:
        # Run a custom evaluator and save results
        if self.evaluators and self.accelerator.is_local_main_process:
            if self.accelerator.num_processes != 1:
                logger.warning(
                    "Custom evaluator can be run with this Trainer only when a single accelerator process is running."
                )
                return {}

            run_dir = self._get_output_dir(trial=trial)
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            output_dir = os.path.join(run_dir, checkpoint_folder, "evals")
            os.makedirs(output_dir, exist_ok=True)
            eval_metrics = {}
            for _, evaluator in self.evaluators.items():
                eval_args = {
                    "output_dir": output_dir,
                    "template_args": self.template_args,
                    "model": self.model,
                    "tokenizer": self.processing_class,
                }
                eval_metrics.update(evaluator.evaluate(**eval_args))
            self.log(eval_metrics)
            return eval_metrics

        if eval_dataset is None or eval_dataset == _EVAL_PLACEHOLDER:
            return {}
        # Run the default HF Trainer evaluate method when eval dataset is provided
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    def _sanitize_log_value(self, value: Any) -> Any:
        if torch.is_tensor(value):
            value = value.detach().cpu()
            return float(value.item()) if value.numel() == 1 else value.tolist()
        if isinstance(value, dict):
            return {k: self._sanitize_log_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._sanitize_log_value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._sanitize_log_value(v) for v in value)
        # NumPy scalars expose item(); keep native scalars unchanged.
        if hasattr(value, "item") and callable(getattr(value, "item")):
            try:
                return value.item()
            except Exception:
                pass
        return value

    def log(self, logs: Dict[str, Any], *args, **kwargs):
        safe_logs = self._sanitize_log_value(logs)
        return super().log(safe_logs, *args, **kwargs)
