import torch
import scipy as sc
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from evals.metrics.base import unlearning_metric
from evals.metrics.utils import to_host


@unlearning_metric(name="hm_aggregate")
def hm_aggregate(model, **kwargs):
    values = [result["agg_value"] for _, result in kwargs["pre_compute"].items()]
    return {"agg_value": sc.stats.hmean(values)}


@unlearning_metric(name="classifier_prob")
def classifier_prob(model, **kwargs):
    batch_size = kwargs.get("batch_size", 32)
    max_length = kwargs.get("max_length", 512)
    class_id = kwargs.get("class_id", 0)
    text_key = kwargs.get("text_key", "generation")
    classifier_model_args = kwargs["classifier_model_args"]
    classifier_tokenization_args = kwargs["classifier_tokenization_args"]
    device = kwargs.get("device", "cuda")

    tokenizer = AutoTokenizer.from_pretrained(**classifier_tokenization_args)
    classifier = AutoModelForSequenceClassification.from_pretrained(
        **classifier_model_args
    ).to(device)

    data = kwargs["pre_compute"]["text"]["value_by_index"]
    data_list = [
        {"text": entry[text_key], "index": int(key)} for key, entry in data.items()
    ]

    # Create DataLoader
    dataloader = DataLoader(
        data_list,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=bool(torch.cuda.is_available()),
        persistent_workers=False,
    )

    scores_by_index = {}
    for batch in tqdm(dataloader):
        batch_texts = batch["text"]
        batch_indices = batch["index"].tolist()

        # Tokenize the batch of texts
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Run the classifier
        with torch.inference_mode():
            outputs = classifier(**inputs)
        # Convert logits to probabilities
        scores = F.softmax(outputs.logits, dim=-1)[:, class_id].float()

        # Map predictions to labels
        for idx, prob, text in zip(batch_indices, scores, batch_texts):
            # Add the prediction to the original data
            scores_by_index[idx] = {"score": to_host(prob), text_key: text}
    class_scores = torch.stack(
        [
            torch.as_tensor(to_host(evals["score"]))
            for evals in scores_by_index.values()
            if evals["score"] is not None
        ]
    )
    return {"agg_value": float(class_scores.mean().item()), "value_by_index": scores_by_index}
