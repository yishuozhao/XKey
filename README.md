# XKey Demo

This is the compact GitHub upload folder for the XKey camera-ready demo. It
shows the ECG key generation pipeline: processed ECG windows -> LSTM feature
encoder -> quantization -> compressed-sensing-style reconciliation -> privacy
amplification.

This demo is runnable on CPU. It does not include private biometric recordings,
participant information, full training data, or the full multi-modality
benchmark code.

## Files

```text
XKey_GitHub/
  README.md
  LICENSE
  requirements.txt
  .gitignore
  PUBLIC_DATASETS.md
  ecg_demo.yaml
  demo_ecg_keygen.py
  xkey_ecg.py
  test_ecg_demo.py
  sample_data/
    mit_bih_demo_pairs.npy
    README.md
  checkpoints/
    ecg_lstm_5bit_demo.pth
    README.md
```

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python demo_ecg_keygen.py --config ecg_demo.yaml
```

Expected output:

```text
XKey ECG Demo
Processed ECG windows: 26
Device: cpu
Device A raw key length: 128 bits
Device B raw key length: 128 bits
Bit agreement before reconciliation: 96.88%
Reconciliation success: True
Corrected raw bit errors: 4
Residual raw bit errors: 0
Final key match after reconciliation: True
Final key length: 128 bits
```

## Test

```bash
python -m pytest -q
```

## Data Notice

The included ECG sample is a small processed subset for demonstrating the
pipeline. Self-collected biometric data and restricted datasets are not
released due to privacy and ethical restrictions.

## Citation

```bibtex
@article{xkey2026,
...
}
```

## Code Availability Sentence

To facilitate reproducibility, we provide the ECG demo implementation,
configuration files, and instructions for running the XKey key generation
pipeline at `https://github.com/yishuozhao/XKey`; due to privacy and ethical
restrictions, the self-collected biometric data are not publicly released.
