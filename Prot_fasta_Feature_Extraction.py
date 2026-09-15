"""ProtT5 residue feature extraction used by RPCG-EDL."""

import os
import re

import torch
from transformers import T5EncoderModel, T5Tokenizer


_tokenizer = None
_model = None
_device = None


def _load_prott5():
    """Load ProtT5 once, on first use, instead of during module import."""
    global _tokenizer, _model, _device
    if _model is not None:
        return _tokenizer, _model, _device

    model_path = os.environ.get("PROTT5_PATH")
    if not model_path:
        raise FileNotFoundError("Set PROTT5_PATH to the local ProtT5 model directory before running RPCG-EDL.")
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"ProtT5 model directory not found: {model_path}. "
            "Set the PROTT5_PATH environment variable before running RPCG-EDL."
        )

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
    _model = T5EncoderModel.from_pretrained(model_path).to(_device).eval()
    if torch.cuda.is_available():
        _model = _model.half()

    parameter_count = sum(parameter.numel() for parameter in _model.parameters()) // 1_000_000
    print(f"Loaded ProtT5 from {model_path} ({parameter_count}M parameters) on {_device}.")
    return _tokenizer, _model, _device


def embed_dataset(sequence, shift_left=0, shift_right=-1):
    """Return per-residue ProtT5 embeddings for one protein sequence."""
    tokenizer, model, device = _load_prott5()
    sequence = re.sub(r"[UZOB]", "X", "".join(sequence.split()))
    residues = list(sequence)

    with torch.no_grad():
        tokenized = tokenizer.batch_encode_plus(
            [residues],
            add_special_tokens=True,
            padding=True,
            is_split_into_words=True,
            return_tensors="pt",
        )
        embedding = model(input_ids=tokenized["input_ids"].to(device))[0][0]

    return embedding.detach().cpu().numpy()[shift_left:shift_right]
