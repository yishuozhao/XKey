# XKey

This is the compact GitHub upload folder for the XKey camera-ready demo. It
shows the ECG key generation pipeline: processed ECG windows -> LSTM feature
encoder -> quantization -> compressed-sensing-style reconciliation -> privacy
amplification.

This code is runnable on CPU. It does not include private biometric recordings,
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
Bit agreement before reconciliation: xx.xx%
Reconciliation success: True
Corrected raw bit errors: x
Residual raw bit errors: x
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
author = {Zhao, Yishuo and Li, Xiaoyang and Lin, Qi and Xu, Weitao and Hu, Pengfei and Shen, Yiran},
title = {XKey: A Unified Deep Learning Framework for Wearable-based Key Generation},
year = {2026},
issue_date = {September 2026},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {10},
number = {3},
url = {https://doi.org/10.1145/3831641},
doi = {10.1145/3831641},
abstract = {Secure key generation from shared biometric signals has emerged as a lightweight alternative to traditional cryptographic key distribution, particularly for resource-constrained wearable devices. However, existing methods are often modality-specific and rely heavily on handcrafted features, which limit their generalizability and efficiency across diverse wearable-based applications. In this paper, we propose XKey, a unified framework that formulates feature extraction from biometric signals as an optimization problem that jointly maximizes inter-device similarity and key entropy. XKey employs deep sequence models to automatically learn robust, information-rich representations, facilitating reliable key generation tailored to biometric signals. To further improve agreement rates, we incorporate a flexible reconciliation mechanism based on compressed sensing. We demonstrate XKey's generalizability by implementing it for distributed key generation using cardiac-based and gait-based signals. Extensive evaluations on cardiac and gait datasets show that XKey achieves high key agreement rates up to 100.0\% and faster key generation rate compared to state-of-the-art works (with improvements of 12.5\% for cardiac signals and 27.8\% for gait signals). These results underscore XKey's promise as a practical and scalable solution for secure communication between current commercial wearable devices. We release the XKey code and instructions at https://github.com/yishuozhao/XKey.},
journal = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.},
month = sep,
articleno = {195},
numpages = {30},
keywords = {Wearable devices, Key generation, Deep learning}
}
```

## Code Availability Sentence

To facilitate reproducibility, we provide the ECG implementation,
configuration files, and instructions for running the XKey key generation
pipeline at `https://github.com/yishuozhao/XKey`; due to privacy and ethical
restrictions, the self-collected biometric data are not publicly released.
