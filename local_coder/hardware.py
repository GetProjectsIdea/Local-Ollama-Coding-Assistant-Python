"""Hardware check and model advice: which open-weight model fits your RAM?

Download sizes come from the Ollama library and Hugging Face model pages (checked
November 2026). "Needs" is a practical total-RAM guide that leaves room for the
operating system, a browser and VS Code, not a hard limit.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess

# name, download size (GB), practical RAM needed (GB), what it is good for, how to get it
MODEL_CATALOG = [
    {"model": "qwen2.5-coder:1.5b", "family": "Qwen2.5-Coder", "download_gb": 0.99, "needs_gb": 4,
     "good_for": "very old laptops; short explanations and small snippets",
     "get": "ollama pull qwen2.5-coder:1.5b"},
    {"model": "qwen2.5-coder:3b", "family": "Qwen2.5-Coder", "download_gb": 1.9, "needs_gb": 6,
     "good_for": "best all-round pick for 8 GB laptops: explain, fix, docstrings, tests",
     "get": "ollama pull qwen2.5-coder:3b"},
    {"model": "hf.co/unsloth/gemma-4-E2B-it-GGUF:Q4_K_M", "family": "Gemma 4 E2B", "download_gb": 3.1,
     "needs_gb": 8, "good_for": "Gemma 4 E2B as a text-only 4-bit GGUF; good Hindi explanations",
     "get": "ollama pull hf.co/unsloth/gemma-4-E2B-it-GGUF:Q4_K_M"},
    {"model": "gemma4:e2b", "family": "Gemma 4 E2B", "download_gb": 7.2, "needs_gb": 12,
     "good_for": "official Ollama Gemma 4 E2B with image input",
     "get": "ollama pull gemma4:e2b"},
    {"model": "qwen2.5-coder:7b", "family": "Qwen2.5-Coder", "download_gb": 4.7, "needs_gb": 12,
     "good_for": "stronger code generation on 16 GB laptops",
     "get": "ollama pull qwen2.5-coder:7b"},
    {"model": "gemma4:12b", "family": "Gemma 4 12B", "download_gb": 7.6, "needs_gb": 14,
     "good_for": "16 GB laptops; better reasoning, 256K context",
     "get": "ollama pull gemma4:12b"},
    {"model": "gemma4:e4b", "family": "Gemma 4 E4B", "download_gb": 9.6, "needs_gb": 16,
     "good_for": "default gemma4 tag (gemma4:latest)",
     "get": "ollama pull gemma4:e4b"},
    {"model": "gemma4:26b", "family": "Gemma 4 26B MoE", "download_gb": 18, "needs_gb": 24,
     "good_for": "workstations; MoE with about 4B active parameters",
     "get": "ollama pull gemma4:26b"},
    {"model": "flint", "family": "Recursive Flint 40B-A2.5B MoE", "download_gb": 23, "needs_gb": 32,
     "good_for": "lab PCs / 32 GB+ machines; strongest local coding model in this list",
     "get": "ollama pull flint"},
]

PREFERRED = ["qwen2.5-coder:1.5b", "qwen2.5-coder:3b", "qwen2.5-coder:7b", "gemma4:12b", "flint"]

TIERS = [
    (0, "under 6 GB", "Use the smallest model and close Chrome while it answers. Or use the mock backend to learn the app."),
    (6, "6-10 GB (typical 8 GB student laptop)", "A 1.5B-3B coder model at 4-bit runs on the CPU. Keep num_ctx at 4096."),
    (10, "10-20 GB (16 GB laptop)", "7B-12B models fit. Gemma 4 12B or Qwen2.5-Coder 7B are good upgrades."),
    (20, "20-40 GB", "26B MoE models fit; Flint needs about 26 GB at 4-bit, so 32 GB is the practical minimum."),
    (40, "40 GB and more", "Everything in the list fits, including Flint at higher precision."),
]


def total_ram_gb() -> float:
    """Total physical memory in GB (Linux, macOS, Windows), or 0.0 if unknown."""
    try:
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            pages, size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and size > 0:
                return round(pages * size / 1e9, 1)
    except (ValueError, OSError):
        pass
    system = platform.system()
    try:
        if system == "Darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
            return round(int(out.stdout.strip()) / 1e9, 1)
        if system == "Windows":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
            return round(stat.ullTotalPhys / 1e9, 1)
    except Exception:  # noqa: BLE001 - best effort only
        pass
    return 0.0


def available_ram_gb() -> float:
    """Free + reclaimable memory on Linux (MemAvailable). 0.0 elsewhere."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return round(int(line.split()[1]) * 1024 / 1e9, 1)
    except OSError:
        pass
    return 0.0


def gpu_hint() -> str:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "Apple Silicon (unified memory, Metal)"
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip():
                return "NVIDIA " + out.stdout.strip().splitlines()[0]
        except (OSError, subprocess.SubprocessError):
            pass
        return "NVIDIA GPU (nvidia-smi found)"
    return "none detected (CPU only - that is fine for 1.5B-3B models)"


def tier_for(ram_gb: float) -> dict:
    chosen = TIERS[0]
    for tier in TIERS:
        if ram_gb >= tier[0]:
            chosen = tier
    return {"label": chosen[1], "advice": chosen[2]}


def recommend(ram_gb: float) -> dict:
    """Split the catalogue into models that fit comfortably, tight fits and too-big ones."""
    fits, tight, too_big = [], [], []
    for m in MODEL_CATALOG:
        if ram_gb >= m["needs_gb"]:
            fits.append(m)
        elif ram_gb >= m["download_gb"] + 2:
            tight.append(m)
        else:
            too_big.append(m)
    # for coding, prefer this upgrade path; pick the biggest one that fits comfortably
    fit_names = {m["model"] for m in fits}
    best = PREFERRED[0]
    for name in PREFERRED:
        if name in fit_names:
            best = name
    return {"ram_gb": ram_gb, "tier": tier_for(ram_gb), "best": best,
            "fits": fits, "tight": tight, "too_big": too_big}


def suggest_num_ctx(ram_gb: float) -> int:
    if ram_gb and ram_gb < 10:
        return 4096
    if ram_gb < 20:
        return 8192
    return 16384


def report() -> dict:
    ram = total_ram_gb()
    return {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "cpus": os.cpu_count() or 0,
        "ram_gb": ram,
        "available_gb": available_ram_gb(),
        "gpu": gpu_hint(),
        "suggested_num_ctx": suggest_num_ctx(ram),
        **{k: v for k, v in recommend(ram).items() if k != "ram_gb"},
    }
