import os
import json
import logging
import csv
import torch
from evals.metrics import get_metrics

logger = logging.getLogger("evaluator")


class Evaluator:
    def __init__(self, name, eval_cfg, **kwargs):
        self.name = name
        self.eval_cfg = eval_cfg
        self.metrics_cfg = self.eval_cfg.metrics
        self.metrics = self.load_metrics(self.metrics_cfg)
        logger.info(
            f"Evaluations stored in the experiment directory: {self.eval_cfg.output_dir}"
        )

    def get_logs_file_path(self, output_dir, suffix="EVAL"):
        """Returns the path to json file to store results"""
        logs_filename = os.path.join(output_dir, f"{self.name}_{suffix}.json")
        return logs_filename

    def load_logs_from_file(self, file):
        """Returns the cache of existing results"""
        logs = {}
        if os.path.exists(file):
            logger.info(f"Loading existing evaluations from {file}")
            with open(file, "r") as f:
                logs = json.load(f)
        return logs

    def save_logs(self, logs, file):
        """Save the logs in a json file"""
        logs = dict(sorted(logs.items()))
        os.makedirs(os.path.dirname(file), exist_ok=True)
        try:
            with open(file, "w") as f:
                json.dump(logs, f, indent=4)
        except Exception as e:
            raise RuntimeError(f"Failed to save {file}: {e}")

    def _is_numeric(self, value):
        return (
            isinstance(value, (int, float)) and not isinstance(value, bool)
        ) or (torch.is_tensor(value) and value.numel() == 1)

    def _to_python_number(self, value):
        if torch.is_tensor(value):
            return float(value.detach().item())
        return float(value)

    def _flatten_metric_rows(self, metric_name, metric_results):
        rows = []

        if not isinstance(metric_results, dict):
            return rows

        agg_value = metric_results.get("agg_value", None)
        subset_keys = [
            key
            for key in metric_results.keys()
            if key not in ("agg_value", "value_by_index")
        ]

        def collect_rows(subset_name, value_by_index):
            for example_id, example_values in value_by_index.items():
                if example_values is None:
                    rows.append(
                        {
                            "metric": metric_name,
                            "subset": subset_name,
                            "example_id": example_id,
                            "stat_key": "value",
                            "value": None,
                            "agg_value": agg_value,
                        }
                    )
                    continue

                if self._is_numeric(example_values):
                    rows.append(
                        {
                            "metric": metric_name,
                            "subset": subset_name,
                            "example_id": example_id,
                            "stat_key": "value",
                            "value": self._to_python_number(example_values),
                            "agg_value": agg_value,
                        }
                    )
                    continue

                if isinstance(example_values, dict):
                    emitted = False
                    for stat_key, stat_value in example_values.items():
                        if self._is_numeric(stat_value):
                            emitted = True
                            rows.append(
                                {
                                    "metric": metric_name,
                                    "subset": subset_name,
                                    "example_id": example_id,
                                    "stat_key": stat_key,
                                    "value": self._to_python_number(stat_value),
                                    "agg_value": agg_value,
                                }
                            )
                    if not emitted:
                        rows.append(
                            {
                                "metric": metric_name,
                                "subset": subset_name,
                                "example_id": example_id,
                                "stat_key": "value",
                                "value": None,
                                "agg_value": agg_value,
                            }
                        )

        if "value_by_index" in metric_results:
            collect_rows("default", metric_results["value_by_index"])

        for subset_name in subset_keys:
            subset_results = metric_results.get(subset_name, {})
            if isinstance(subset_results, dict) and "value_by_index" in subset_results:
                collect_rows(subset_name, subset_results["value_by_index"])

        return rows

    def save_detailed_logs(self, logs, file):
        """Save per-example metric rows to a CSV file."""
        rows = []
        for metric_name, metric_results in sorted(logs.items()):
            rows.extend(self._flatten_metric_rows(metric_name, metric_results))

        os.makedirs(os.path.dirname(file), exist_ok=True)
        fieldnames = ["metric", "subset", "example_id", "stat_key", "value", "agg_value"]
        with open(file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def prepare_model(self, model):
        """Prepare model for evaluation"""
        model.eval()
        # Move model to GPU if available
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            model = model.to(device)
            logger.info(f"Model moved to GPU: {device}")
        else:
            logger.warning("CUDA not available, model will run on CPU (gradnorm will be very slow)")
        return model

    def load_metrics(self, metrics_cfg):
        """Load metrics for evaluation"""
        metrics = get_metrics(metrics_cfg)
        return metrics

    def summarize(self, logs):
        """Summarize the metrics results"""
        metric_summary = {}
        for metric_name, metric_results in logs.items():
            if metric_name not in self.metrics:
                continue
            agg_value = metric_results.get("agg_value", None)
            if agg_value is not None:
                metric_summary[metric_name] = self._to_python_number(agg_value)
        return metric_summary

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        # set flag to overwrite metrics
        overwrite = self.eval_cfg.overwrite if overwrite is None else overwrite

        # Prepare model for evaluation
        model = self.prepare_model(model)

        # Set output_dir and file to store results
        output_dir = output_dir if output_dir else self.eval_cfg.output_dir
        logs_file_path = self.get_logs_file_path(output_dir)
        detailed_logs_file_path = os.path.join(output_dir, f"{self.name}_EVAL_DETAILS.csv")
        summary_file_path = self.get_logs_file_path(output_dir, suffix="SUMMARY")

        # Load existing results from file if any.
        logs = self.load_logs_from_file(logs_file_path) if not overwrite else {}

        logger.info(f"***** Running {self.name} evaluation suite *****")
        logger.info(f"Fine-grained evaluations will be saved to: {logs_file_path}")
        logger.info(
            f"Aggregated evaluations will be summarised in: {summary_file_path}"
        )
        for metric_name, metric_fn in self.metrics.items():
            if not overwrite and metric_name in logs and logs[metric_name]:
                logger.info(f"Skipping {metric_name}, already evaluated.")
                if "agg_value" in logs[metric_name]:
                    logger.info(
                        f"Result for metric {metric_name}:\t{logs[metric_name]['agg_value']}"
                    )
                self.save_logs(self.summarize(logs), summary_file_path)
                continue
            _ = logs.pop(metric_name, None)  # overwriting existing evals if present
            kwargs = {
                "tokenizer": kwargs.get("tokenizer", None),
                "template_args": kwargs.get("template_args", None),
                "output_dir": output_dir,
            }
            metrics_args = self.eval_cfg.metrics[metric_name]
            _
            result = metric_fn(
                model,
                metric_name=metric_name,
                cache=logs,
                **kwargs,
                **metrics_args,
            )
            if "agg_value" in result:
                logger.info(f"Result for metric {metric_name}:\t{result['agg_value']}")
            self.save_logs(logs, logs_file_path)
            self.save_detailed_logs(logs, detailed_logs_file_path)
            self.save_logs(self.summarize(logs), summary_file_path)

        return self.summarize(logs)
