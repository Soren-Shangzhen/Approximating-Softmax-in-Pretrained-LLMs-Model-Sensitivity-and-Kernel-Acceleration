"""Evaluation token stream: WikiText-103 (raw) test split, joined and tokenized once per tokenizer.

The records of the test split are joined with "\\n\\n" and tokenized as one text; the first ``n_tokens`` ids
(200,000 in the paper) are cached as ``tokens_wikitext_<tag>_<n_tokens>.pt`` in the working directory.  At a
sequence length of 2048 the stream gives 97 aligned, non-overlapping blocks (App. A.2).  Every run records a
``token_signature`` of the stream, and the pipeline checks it against reference results before reusing them.
"""
import hashlib
import math
import os
import re

import torch
from datasets import load_dataset

DATASETS = {
    "wikitext": dict(repo="Salesforce/wikitext", cfg="wikitext-103-raw-v1", split="test", field="text"),
}


def _gather_text(name):
    spec = DATASETS[name]
    ds = load_dataset(spec["repo"], spec["cfg"], split=spec["split"])
    return "\n\n".join(ds[spec["field"]])


def token_signature(ids):
    """Short fingerprint of a token sequence."""
    return hashlib.md5(ids.numpy().tobytes()).hexdigest()[:12]


def build_tokens(tok, n_tokens=200_000, cache=None, tag=None, dataset="wikitext", verbose=True):
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset}; available: {list(DATASETS)}")
    if cache is None:
        if tag is None:
            tag = getattr(tok, "name_or_path", "unk")
        tag = re.sub(r"[^A-Za-z0-9._-]", "_", str(tag))
        cache = f"tokens_{dataset}_{tag}_{n_tokens}.pt"

    if os.path.exists(cache):
        ids = torch.load(cache, weights_only=True)
        if verbose:
            print(f"[tokens] dataset={dataset} tokens={len(ids)} cache={cache} sig={token_signature(ids)}")
        return ids

    text = _gather_text(dataset)
    ids = tok(text, return_tensors="pt").input_ids[0][:n_tokens]
    if len(ids) < n_tokens:
        raise RuntimeError(f"{dataset} gave only {len(ids)} tokens, fewer than the requested {n_tokens}")
    torch.save(ids, cache)
    if verbose:
        print(f"[tokens] dataset={dataset} tokens={len(ids)} cache={cache} sig={token_signature(ids)} (new)")
    return ids


@torch.no_grad()
def ppl(model, ids, seq_len=2048, device="cuda"):
    """Perplexity over aligned, non-overlapping blocks of seq_len tokens (model's own loss)."""
    nll, ntok = 0.0, 0
    for i in range(0, len(ids) - seq_len, seq_len):
        chunk = ids[i:i + seq_len].unsqueeze(0).to(device)
        loss = model(chunk, labels=chunk).loss
        nll += loss.item() * (seq_len - 1)
        ntok += seq_len - 1
    return math.exp(nll / ntok)
