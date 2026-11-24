import pytest

from local_coder import hardware
from local_coder.hardware import MODEL_CATALOG, recommend, suggest_num_ctx, tier_for


def test_catalog_is_sane():
    names = [m["model"] for m in MODEL_CATALOG]
    assert len(names) == len(set(names))
    for m in MODEL_CATALOG:
        assert m["needs_gb"] > m["download_gb"]
        assert m["get"].startswith("ollama pull ")
    assert "flint" in names and "gemma4:e2b" in names and "qwen2.5-coder:3b" in names


@pytest.mark.parametrize("ram,label", [(4, "under 6 GB"), (8, "6-10 GB"), (16, "10-20 GB"), (32, "20-40 GB"), (64, "40 GB")])
def test_tiers(ram, label):
    assert tier_for(ram)["label"].startswith(label)


def test_8gb_laptop_gets_small_coder():
    r = recommend(8)
    assert r["best"] == "qwen2.5-coder:3b"
    fit = {m["model"] for m in r["fits"]}
    assert "hf.co/unsloth/gemma-4-E2B-it-GGUF:Q4_K_M" in fit
    assert "flint" in {m["model"] for m in r["too_big"]}


def test_best_pick_path():
    assert recommend(12)["best"] == "qwen2.5-coder:7b"
    assert recommend(16)["best"] == "gemma4:12b"
    assert recommend(40)["best"] == "flint"


def test_16gb_and_32gb():
    assert "gemma4:12b" in {m["model"] for m in recommend(16)["fits"]}
    assert "flint" not in {m["model"] for m in recommend(16)["fits"]}
    assert "flint" in {m["model"] for m in recommend(32)["fits"]}


def test_tiny_machine():
    r = recommend(3)
    assert r["best"] == "qwen2.5-coder:1.5b"
    assert not r["fits"]


def test_num_ctx_suggestion():
    assert suggest_num_ctx(8) == 4096
    assert suggest_num_ctx(16) == 8192
    assert suggest_num_ctx(64) == 16384


def test_report_shape():
    r = hardware.report()
    for key in ("os", "cpus", "ram_gb", "gpu", "tier", "best", "fits", "suggested_num_ctx"):
        assert key in r
    assert r["ram_gb"] >= 0


def test_ram_detection_positive_on_this_machine():
    assert hardware.total_ram_gb() > 0
