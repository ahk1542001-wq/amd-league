#!/usr/bin/env python3
"""MC4 smoke test for the AMD hackathon pool (Radeon Pro W7900D, ROCm).

Stages:
  A  env: torch / GPU
  B  download main model Qwen/Qwen3.8-27B-FP8 -> /root/models (overlay disk)
  C  vllm presence; if missing, pip install vllm, then VERIFY torch stayed ROCm
  D  vLLM load with the scaffold's exact params + one tiny generation
  E  decider Qwen/Qwen3.5-4B transformers load (bf16) + one tiny generation

Prints STAGE-X PASS/FAIL markers. Never raises out; always ends with SMOKE-DONE.
"""
import os, sys, time, traceback

os.environ.setdefault("HF_HOME", "/root/hf")
MODELS = "/root/models"
MAIN_REPO = "Qwen/Qwen3.8-27B-FP8"
DECIDER_REPO = "Qwen/Qwen3.5-4B"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stage(name, fn):
    log(f"STAGE-{name} START")
    t = time.time()
    try:
        fn()
    except Exception:
        log(f"STAGE-{name} FAIL ({time.time()-t:.0f}s)")
        traceback.print_exc()
        return False
    log(f"STAGE-{name} PASS ({time.time()-t:.0f}s)")
    return True


def a_env():
    import torch
    log(f"torch={torch.__version__} avail={torch.cuda.is_available()}")
    p = torch.cuda.get_device_properties(0)
    log(f"gpu={p.name} vram_gb={p.total_memory/1e9:.1f}")


def b_download_main():
    from huggingface_hub import snapshot_download
    out = os.path.join(MODELS, "Qwen3.8-27B-FP8")
    snapshot_download(repo_id=MAIN_REPO, local_dir=out,
                      ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx"])
    total = sum(f.stat().st_size for f in __import__("pathlib").Path(out).rglob("*") if f.is_file())
    log(f"main model downloaded: {total/1e9:.1f} GB at {out}")


def c_vllm():
    try:
        import vllm
        log(f"vllm already installed: {vllm.__version__}")
        return
    except ImportError:
        log("vllm missing -> pip install vllm (this can take several minutes)")
    import subprocess
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "vllm"],
                       capture_output=True, text=True)
    log("pip tail: " + "\n".join((r.stdout + r.stderr).splitlines()[-5:]))
    if r.returncode != 0:
        raise RuntimeError(f"pip install vllm failed rc={r.returncode}")
    import torch
    v = torch.__version__
    log(f"torch after vllm install: {v}")
    assert "rocm" in v.lower(), f"FATAL torch swapped to non-ROCm build: {v}"
    import vllm
    log(f"vllm installed: {vllm.__version__}")


def d_main_load():
    from vllm import LLM, SamplingParams
    llm = LLM(model=os.path.join(MODELS, "Qwen3.8-27B-FP8"),
              dtype="fp8", gpu_memory_utilization=0.70,
              max_model_len=16384, enforce_eager=False)
    out = llm.generate(["The capital of France is"],
                       SamplingParams(temperature=0.0, max_tokens=20))
    log(f"main generate -> {out[0].outputs[0].text!r}")


def e_decider_load():
    from huggingface_hub import snapshot_download
    out = os.path.join(MODELS, "Qwen3.5-4B")
    snapshot_download(repo_id=DECIDER_REPO, local_dir=out,
                      ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx"])
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(out)
    model = AutoModelForCausalLM.from_pretrained(
        out, torch_dtype=torch.bfloat16, device_map="cuda")
    ids = tok("The capital of France is", return_tensors="pt").to("cuda")
    gen = model.generate(**ids, max_new_tokens=20, do_sample=False)
    log(f"decider generate -> {tok.decode(gen[0], skip_special_tokens=True)!r}")


if __name__ == "__main__":
    os.makedirs(MODELS, exist_ok=True)
    ok_a = stage("A", a_env)
    ok_b = stage("B", b_download_main)
    ok_c = stage("C", c_vllm)
    if ok_b and ok_c:
        stage("D", d_main_load)
    else:
        log("STAGE-D SKIPPED (needs B and C)")
    stage("E", e_decider_load)
    log("SMOKE-DONE")
