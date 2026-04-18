"""
Enum class for attacks. Also contains the base attack class.
"""

from enum import Enum
import json
import os
from itertools import islice
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import logging

logger = logging.getLogger(__name__)


# Attack definitions
class AllAttacks(str, Enum):
    LOSS = "loss"
    REFERENCE_BASED = "ref"
    ZLIB = "zlib"
    MIN_K = "min_k"
    MIN_K_PLUS_PLUS = "min_k++"
    GRADNORM = "gradnorm"
    RECALL = "recall"


# Base attack class
class Attack:
    def __init__(self, model, data, collator, batch_size, **kwargs):
        """Initialize attack with model and create dataloader."""
        self.model = model
        pin_memory = bool(torch.cuda.is_available())
        self.dataloader = DataLoader(
            data,
            batch_size=batch_size,
            collate_fn=collator,
            num_workers=0,
            pin_memory=pin_memory,
            persistent_workers=False,
        )
        self.checkpoint_dir = kwargs.pop("checkpoint_dir", None)
        self.checkpoint_every_n_batches = kwargs.pop("checkpoint_every_n_batches", 100)
        self.setup(**kwargs)

    def setup(self, **kwargs):
        """Setup attack-specific parameters."""
        pass

    def compute_batch_values(self, batch):
        """Process a batch through model to get needed statistics."""
        raise NotImplementedError

    def compute_score(self, sample_stats):
        """Compute MIA score for a single sample."""
        raise NotImplementedError

    def _score_to_tensor(self, score):
        if isinstance(score, torch.Tensor):
            return score.detach().to(dtype=torch.float32, device="cpu")
        return torch.tensor(score, dtype=torch.float32, device="cpu")

    def _score_to_python(self, score):
        if isinstance(score, torch.Tensor):
            return float(score.detach().item())
        return float(score)

    def attack(self):
        """Run full MIA attack."""
        all_scores = []
        all_indices = []

        start_batch_idx = 0
        checkpoint_file = None
        if self.checkpoint_dir is not None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            checkpoint_file = os.path.join(self.checkpoint_dir, "attack_checkpoint.json")
            if os.path.exists(checkpoint_file):
                checkpoint = None
                try:
                    with open(checkpoint_file, "r") as f:
                        checkpoint = json.load(f)
                except (json.JSONDecodeError, OSError, ValueError):
                    logger.warning(
                        "[Attack] Corrupted checkpoint at %s; restarting this attack from scratch.",
                        checkpoint_file,
                    )
                    try:
                        os.remove(checkpoint_file)
                    except OSError:
                        pass
                    checkpoint = None

                if checkpoint is None:
                    checkpoint = {}
                if checkpoint.get("done"):
                    logger.info(f"[Attack] Loaded completed checkpoint from {checkpoint_file}")
                    return {
                        "agg_value": float(checkpoint["agg_value"]),
                        "value_by_index": checkpoint["value_by_index"],
                    }
                start_batch_idx = int(checkpoint.get("next_batch_idx", 0))
                all_scores = [
                    torch.tensor(score, dtype=torch.float32, device="cpu")
                    for score in checkpoint.get("all_scores", [])
                ]
                all_indices = list(checkpoint.get("all_indices", []))
                logger.info(
                    f"[Attack] Resuming from checkpoint {checkpoint_file} at batch {start_batch_idx}"
                )
        
        dataset = self.dataloader.dataset
        fallback_indices = list(range(len(dataset)))
        if hasattr(dataset, "data") and hasattr(dataset.data, "column_names") and "index" in dataset.data.column_names:
            fallback_indices = [int(idx) for idx in dataset.data["index"]]

        fallback_cursor = start_batch_idx * self.dataloader.batch_size

        for batch_idx, batch in enumerate(
            tqdm(
                islice(self.dataloader, start_batch_idx, None),
                total=len(self.dataloader) - start_batch_idx,
            ),
            start=start_batch_idx,
        ):
            if "input_ids" not in batch:
                if "0" in batch and isinstance(batch["0"], dict) and "input_ids" in batch["0"]:
                    batch = dict(batch["0"])
                else:
                    first_val = next(iter(batch.values()))
                    if isinstance(first_val, dict) and "input_ids" in first_val:
                        batch = dict(first_val)
                    else:
                        raise KeyError(f"Expected 'input_ids' in batch, but got missing. Batch keys: {batch.keys()}")
            
            current_batch_size = len(batch["input_ids"])
            guessed_indices = fallback_indices[fallback_cursor : fallback_cursor + current_batch_size]
            fallback_cursor += current_batch_size

            if "index" in batch:
                indices = batch.pop("index").tolist()
            else:
                indices = guessed_indices
            
            batch_values = self.compute_batch_values(batch)
            scores = [self._score_to_tensor(self.compute_score(values)) for values in batch_values]

            all_scores.extend(scores)
            all_indices.extend(indices)

            if checkpoint_file is not None and (
                (batch_idx + 1) % self.checkpoint_every_n_batches == 0
                or batch_idx + 1 == len(self.dataloader)
            ):
                serialized_scores = [self._score_to_python(score) for score in all_scores]
                scores_by_index = {
                    str(idx): {"score": self._score_to_python(score)}
                    for idx, score in zip(all_indices, all_scores)
                }
                with open(checkpoint_file, "w") as f:
                    json.dump(
                        {
                            "done": False,
                            "next_batch_idx": batch_idx + 1,
                            "all_scores": serialized_scores,
                            "all_indices": [int(index) for index in all_indices],
                            "agg_value": float(torch.stack(all_scores).mean().detach().item()),
                            "value_by_index": scores_by_index,
                        },
                        f,
                        indent=4,
                    )

        serialized_scores = [self._score_to_python(score) for score in all_scores]
        scores_by_index = {
            str(idx): {"score": self._score_to_python(score)}
            for idx, score in zip(all_indices, all_scores)
        }

        agg_value = float(torch.stack(all_scores).mean().detach().item()) if all_scores else None

        if checkpoint_file is not None:
            with open(checkpoint_file, "w") as f:
                json.dump(
                    {
                        "done": True,
                        "next_batch_idx": len(self.dataloader),
                        "all_scores": serialized_scores,
                        "all_indices": [int(index) for index in all_indices],
                        "agg_value": agg_value,
                        "value_by_index": scores_by_index,
                    },
                    f,
                    indent=4,
                )

        return {
            "agg_value": agg_value,
            "value_by_index": scores_by_index,
        }
