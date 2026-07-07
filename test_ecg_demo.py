from xkey_ecg import load_config, run_ecg_demo


def test_ecg_demo_pipeline():
    result = run_ecg_demo(load_config("ecg_demo.yaml"))
    assert result["raw_key_length"] == 128
    assert result["final_key_length"] == 128
    assert result["final_key_match"] is True
    assert result["reconciliation"].residual_errors == 0
