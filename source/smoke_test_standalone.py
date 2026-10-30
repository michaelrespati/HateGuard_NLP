
"""smoke_test_standalone.py — loads artifacts ONLY from files; does not depend on notebook state.
Usage: python smoke_test_standalone.py <artifact_dir> [cpu|cuda]   (exit code 0 = PASS)"""
import sys, os, json, hashlib, importlib.util
import numpy as np, pandas as pd, joblib, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ART = sys.argv[1]; DEV = sys.argv[2] if len(sys.argv) > 2 else "cpu"
TOL = 1e-3   # cross-device numerical tolerance (GPU FP32 vs CPU FP32)
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(c)
    return h.hexdigest()
def fail(msg): print("❌ SMOKE TEST FAILED:", msg); sys.exit(1)

add = json.load(open(os.path.join(ART, "experiment_manifest_addendum.json"), encoding="utf-8"))
man = json.load(open(os.path.join(ART, "experiment_manifest.json"), encoding="utf-8"))
core_path = os.path.join(ART, "hateguard_core.py")
if sha(core_path) != add["core_sha256"]: fail("hateguard_core.py SHA-256 does not match the addendum")
spec = importlib.util.spec_from_file_location("hateguard_core", core_path); core = importlib.util.module_from_spec(spec); sys.modules["hateguard_core"] = core; spec.loader.exec_module(core)
mdir = os.path.join(ART, "model_hate_speech_final")
wf = "model.safetensors" if os.path.isfile(os.path.join(mdir, "model.safetensors")) else "pytorch_model.bin"
if sha(os.path.join(mdir, wf)) != man["trusted_model_sha256"]: fail("Model weights SHA-256 does not match trusted_model_sha256")
bpath = os.path.join(ART, "preprocessing_bundle.joblib")
if sha(bpath) != add["bundle_sha256"]: fail("Bundle SHA-256 does not match the addendum (do not joblib.load an unverified file)")
bundle = joblib.load(bpath); cfg = bundle["config"]; kamus = bundle["kamus_gabungan"]
ph = hashlib.sha256("||".join(core.clean_text(s, kamus) for s in cfg["preprocessing_parity_samples"]).encode("utf-8")).hexdigest()
if ph != cfg["preprocessing_parity_hash"]: fail("Preprocessing parity hash mismatch")
tok = AutoTokenizer.from_pretrained(mdir); model = AutoModelForSequenceClassification.from_pretrained(mdir).to(DEV, dtype=torch.float32).eval()
ref = pd.read_csv(os.path.join(ART, "inference_parity_reference.csv"))
texts = [core.clean_text(t, kamus) for t in ref["Tweet"].astype(str)]
run = lambda: core.softmax_hs(core.logits_fp32(texts, tok, model, torch.device(DEV), cfg["max_length"], cfg["padding_strategy"]))
p1, p2 = run(), run()
if not np.array_equal(p1, p2): fail("Inference is not deterministic")
diff = np.abs(p1 - ref["p_hate_speech"].to_numpy()); thr = cfg["guardrail_thresholds"]
lv_new = np.array([core.prob_to_level(float(p), thr) for p in p1]); lv_ref = ref["guardrail_level"].to_numpy()
near = np.min(np.abs(p1[:, None] - np.array([thr["safe_max"], thr["weak_max"], thr["moderate_max"]])[None, :]), axis=1) < TOL
bad_flips = int(((lv_new != lv_ref) & ~near).sum())
if diff.max() > TOL: fail(f"Maximum probability difference {diff.max():.2e} > {TOL}")
if bad_flips: fail(f"{bad_flips} tier flips outside the threshold tolerance zone")
out = {"status": "PASS", "device": DEV, "n": int(len(ref)), "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
       "tier_flips_within_tolerance_zone": int((lv_new != lv_ref).sum()), "tolerance": TOL}
print("✅ SMOKE TEST PASSED:", json.dumps(out)); json.dump(out, open(os.path.join(ART, "smoke_test_result.json"), "w"))
