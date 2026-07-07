"""Minimal ECG demo pipeline for XKey.

This file keeps the public demo compact: processed ECG pairs go through the
LSTM encoder, projection quantization, lightweight reconciliation, and privacy
amplification.
"""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ECGSequenceModel(nn.Module):
    """LSTM-attention ECG encoder used by the demo checkpoint."""

    def __init__(
        self,
        input_length: int = 512,
        out_bits: int = 5,
        hidden_size: int = 256,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.input_adapter = nn.LayerNorm(input_length)
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.3 if num_layers > 1 else 0.0,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1),
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, out_bits),
        )

    def forward(self, signal_batch: torch.Tensor) -> torch.Tensor:
        x = self.input_adapter(signal_batch).unsqueeze(-1)
        sequence_out, _ = self.lstm(x)
        weights = self.attention(sequence_out)
        context = torch.sum(weights * sequence_out, dim=1)
        return self.fc(context)


def load_model(
    checkpoint_path: str | Path,
    device: torch.device,
    input_length: int = 512,
    out_bits: int = 5,
) -> ECGSequenceModel:
    model = ECGSequenceModel(input_length=input_length, out_bits=out_bits)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def quantize_projection(projections: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    values = np.asarray(projections, dtype=np.float32)
    min_values = values.min(axis=1, keepdims=True)
    max_values = values.max(axis=1, keepdims=True)
    normalized = (values - min_values) / (max_values - min_values + 1e-8)
    return (normalized > threshold).astype(np.uint8)


def bit_agreement(bits_a: np.ndarray, bits_b: np.ndarray) -> float:
    a = np.asarray(bits_a, dtype=np.uint8).reshape(-1)
    b = np.asarray(bits_b, dtype=np.uint8).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("Bit arrays must have the same shape.")
    return 100.0 * (1.0 - np.count_nonzero(a != b) / a.size)


@dataclass(frozen=True)
class ReconciliationResult:
    corrected_bits: np.ndarray
    success: bool
    corrected_errors: int
    residual_errors: int


def _binary_matrix(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    matrix = rng.integers(0, 2, size=(rows, cols), dtype=np.uint8)
    for col in range(cols):
        if not np.any(matrix[:, col]):
            matrix[rng.integers(0, rows), col] = 1
    return matrix


def _syndrome(matrix: np.ndarray, bits: np.ndarray) -> np.ndarray:
    return (matrix @ bits.astype(np.uint8)) % 2


def _decode_sparse_error(
    matrix: np.ndarray,
    target_syndrome: np.ndarray,
    max_errors: int,
) -> tuple[np.ndarray | None, int]:
    cols = matrix.shape[1]
    if not np.any(target_syndrome):
        return np.zeros(cols, dtype=np.uint8), 0

    columns = [matrix[:, i] for i in range(cols)]
    for weight in range(1, max_errors + 1):
        for combo in combinations(range(cols), weight):
            candidate = np.zeros_like(target_syndrome, dtype=np.uint8)
            for idx in combo:
                candidate ^= columns[idx]
            if np.array_equal(candidate, target_syndrome):
                error = np.zeros(cols, dtype=np.uint8)
                error[list(combo)] = 1
                return error, weight
    return None, 0


def reconcile_keys(
    reference_bits: np.ndarray,
    noisy_bits: np.ndarray,
    block_size: int,
    syndrome_rows: int,
    max_errors_per_block: int,
    seed: int,
) -> ReconciliationResult:
    ref = np.asarray(reference_bits, dtype=np.uint8).reshape(-1)
    noisy = np.asarray(noisy_bits, dtype=np.uint8).reshape(-1)
    rng = np.random.default_rng(seed)
    corrected = noisy.copy()
    corrected_errors = 0
    success = True

    for start in range(0, len(ref), block_size):
        end = min(start + block_size, len(ref))
        matrix = _binary_matrix(syndrome_rows, end - start, rng)
        delta = _syndrome(matrix, ref[start:end]) ^ _syndrome(matrix, noisy[start:end])
        error, weight = _decode_sparse_error(matrix, delta, max_errors_per_block)
        if error is None:
            success = False
            continue
        corrected[start:end] ^= error
        corrected_errors += weight

    residual_errors = int(np.count_nonzero(ref != corrected))
    return ReconciliationResult(
        corrected_bits=corrected,
        success=success and residual_errors == 0,
        corrected_errors=int(corrected_errors),
        residual_errors=residual_errors,
    )


def privacy_amplify(bits: np.ndarray, key_length: int, salt: str) -> np.ndarray:
    clean_bits = np.asarray(bits, dtype=np.uint8).reshape(-1) & 1
    payload = np.packbits(clean_bits).tobytes()
    output = bytearray()
    counter = 0
    while len(output) * 8 < key_length:
        digest = hashlib.sha256()
        digest.update(salt.encode("utf-8"))
        digest.update(counter.to_bytes(4, byteorder="big"))
        digest.update(payload)
        output.extend(digest.digest())
        counter += 1
    return np.unpackbits(np.frombuffer(bytes(output), dtype=np.uint8))[:key_length]


def bits_to_hex(bits: np.ndarray) -> str:
    clean_bits = np.asarray(bits, dtype=np.uint8).reshape(-1) & 1
    pad_len = (-clean_bits.size) % 8
    if pad_len:
        clean_bits = np.pad(clean_bits, (0, pad_len), constant_values=0)
    return np.packbits(clean_bits).tobytes().hex()


def run_ecg_demo(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 42))
    set_seed(seed)

    pairs = np.load(config["sample_data"]).astype(np.float32)
    if pairs.ndim != 3 or pairs.shape[1] != 2:
        raise ValueError(f"Expected sample data shape (n_windows, 2, 512), got {pairs.shape}.")

    model_cfg = config["model"]
    device_name = str(model_cfg.get("device", "cpu"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    model = load_model(
        model_cfg["checkpoint"],
        device=device,
        input_length=int(model_cfg.get("input_length", pairs.shape[2])),
        out_bits=int(model_cfg.get("out_bits", 5)),
    )

    num_windows = int(config["feature_extraction"].get("num_windows", len(pairs)))
    raw_key_length = int(config["feature_extraction"].get("raw_key_length", 128))
    pairs = pairs[:num_windows]

    with torch.no_grad():
        signal_a = torch.from_numpy(pairs[:, 0, :]).to(device)
        signal_b = torch.from_numpy(pairs[:, 1, :]).to(device)
        projection_a = model(signal_a).cpu().numpy()
        projection_b = model(signal_b).cpu().numpy()

    threshold = float(config.get("quantization", {}).get("threshold", 0.5))
    raw_bits_a = quantize_projection(projection_a, threshold).reshape(-1)[:raw_key_length]
    raw_bits_b = quantize_projection(projection_b, threshold).reshape(-1)[:raw_key_length]

    rec_cfg = config["reconciliation"]
    reconciliation = reconcile_keys(
        raw_bits_a,
        raw_bits_b,
        block_size=int(rec_cfg.get("block_size", 32)),
        syndrome_rows=int(rec_cfg.get("syndrome_rows", 24)),
        max_errors_per_block=int(rec_cfg.get("max_errors_per_block", 4)),
        seed=int(rec_cfg.get("seed", seed)),
    )

    pa_cfg = config["privacy_amplification"]
    final_key_length = int(pa_cfg.get("final_key_length", raw_key_length))
    salt = str(pa_cfg.get("salt", "xkey-ecg-demo"))
    final_key_a = privacy_amplify(raw_bits_a, final_key_length, salt)
    final_key_b = privacy_amplify(reconciliation.corrected_bits, final_key_length, salt)

    return {
        "num_windows": num_windows,
        "device": str(device),
        "raw_key_length": int(raw_bits_a.size),
        "raw_agreement": bit_agreement(raw_bits_a, raw_bits_b),
        "reconciliation": reconciliation,
        "final_key_match": bool(np.array_equal(final_key_a, final_key_b)),
        "final_key_length": int(final_key_a.size),
        "final_key_hex": bits_to_hex(final_key_a),
    }
