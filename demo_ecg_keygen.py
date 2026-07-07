"""Run the compact XKey ECG key generation demo."""

from __future__ import annotations

import argparse

from xkey_ecg import load_config, run_ecg_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="XKey ECG key generation demo")
    parser.add_argument("--config", default="ecg_demo.yaml")
    args = parser.parse_args()

    result = run_ecg_demo(load_config(args.config))
    reconciliation = result["reconciliation"]

    print("XKey ECG Demo")
    print(f"Processed ECG windows: {result['num_windows']}")
    print(f"Device: {result['device']}")
    print(f"Device A raw key length: {result['raw_key_length']} bits")
    print(f"Device B raw key length: {result['raw_key_length']} bits")
    print(f"Bit agreement before reconciliation: {result['raw_agreement']:.2f}%")
    print(f"Reconciliation success: {reconciliation.success}")
    print(f"Corrected raw bit errors: {reconciliation.corrected_errors}")
    print(f"Residual raw bit errors: {reconciliation.residual_errors}")
    print(f"Final key match after reconciliation: {result['final_key_match']}")
    print(f"Final key length: {result['final_key_length']} bits")
    print(f"Final key preview: {result['final_key_hex'][:32]}")


if __name__ == "__main__":
    main()
