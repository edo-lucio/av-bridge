"""Phase 2 — frozen encoders for image, audio, and text.

The user supplied a registry of vision and audio encoders to run. We
encode all of them so the experiments can be re-run with different
combinations later. Text uses Sentence-Transformers all-MiniLM-L6-v2
per the plan, encoding visual_captions and audio_captions separately
and mean-pooling within each caption pool.

All output matrices are L2-normalised row-wise and saved as float32.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EMB = ROOT / "embeddings"
MANIFEST = DATA / "manifest.csv"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


VISION_ENCODERS = {
    "dinov2-small": {"hf_id": "facebook/dinov2-small", "type": "auto_pooler"},
    "dinov2-base":  {"hf_id": "facebook/dinov2-base",  "type": "auto_pooler"},
    "dinov2-large": {"hf_id": "facebook/dinov2-large", "type": "auto_pooler"},
    "clip-base":    {"hf_id": "openai/clip-vit-base-patch32",  "type": "clip"},
    "clip-large":   {"hf_id": "openai/clip-vit-large-patch14", "type": "clip"},
    "vit-mae-base": {"hf_id": "facebook/vit-mae-base", "type": "mae"},
}

AUDIO_ENCODERS = {
    "clap-unfused": {"hf_id": "laion/clap-htsat-unfused",  "type": "clap", "sr": 48000},
    "clap-fused":   {"hf_id": "laion/clap-htsat-fused",    "type": "clap", "sr": 48000},
    "clap-larger":  {"hf_id": "laion/larger_clap_general", "type": "clap", "sr": 48000},
    "clap-music":   {"hf_id": "laion/larger_clap_music",   "type": "clap", "sr": 48000},
    "mert-95m":     {"hf_id": "m-a-p/MERT-v1-95M",  "type": "mert", "sr": 24000},
    "mert-330m":    {"hf_id": "m-a-p/MERT-v1-330M", "type": "mert", "sr": 24000},
}


def pick_device() -> str:
    """Try real CUDA op; fall back to CPU on any failure (e.g. sm_61 on a new torch)."""
    import os
    if os.environ.get("FORCE_CPU"):
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    try:
        a = torch.randn(64, 64, device="cuda")
        b = torch.randn(64, 64, device="cuda")
        (a @ b).sum().item()
        return "cuda"
    except Exception as e:
        print(f"[warn] CUDA unusable ({e.__class__.__name__}); using CPU")
        return "cpu"


DEVICE = pick_device()


def l2_normalise(M: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n = np.where(n > 1e-12, n, 1.0)
    return (M / n).astype(np.float32)


def load_manifest() -> list[dict]:
    rows: list[dict] = []
    with MANIFEST.open() as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            r["visual_captions"] = json.loads(r["visual_captions"])
            r["audio_captions"] = json.loads(r["audio_captions"])
            rows.append(r)
    return rows


def _resample(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return wav.astype(np.float32)
    import librosa
    return librosa.resample(wav.astype(np.float32), orig_sr=sr_in, target_sr=sr_out)


def _fix_audio(wav: np.ndarray, sr: int, duration_s: float = 10.0) -> np.ndarray:
    """Mono, take a duration_s window centered on the clip midpoint; zero-pad symmetrically if shorter.

    This matches the visual side, which takes the middle frame — both modalities
    summarise the same instant of the clip.
    """
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    target = int(duration_s * sr)
    if len(wav) >= target:
        start = (len(wav) - target) // 2
        return wav[start : start + target].astype(np.float32)
    out = np.zeros(target, dtype=np.float32)
    pad = (target - len(wav)) // 2
    out[pad : pad + len(wav)] = wav.astype(np.float32)
    return out


def encode_vision_auto_pooler(hf_id: str, manifest: list[dict]) -> np.ndarray:
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(hf_id)
    model = AutoModel.from_pretrained(hf_id).to(DEVICE).eval()
    feats = []
    with torch.no_grad():
        for it in manifest:
            img = Image.open(ROOT / it["frame_path"]).convert("RGB")
            x = proc(images=img, return_tensors="pt").to(DEVICE)
            out = model(**x)
            pooled = getattr(out, "pooler_output", None)
            f = pooled if pooled is not None else out.last_hidden_state[:, 0]
            feats.append(f[0].cpu().numpy())
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return np.stack(feats).astype(np.float32)


def encode_vision_clip(hf_id: str, manifest: list[dict]) -> np.ndarray:
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    proc = CLIPImageProcessor.from_pretrained(hf_id)
    model = CLIPVisionModelWithProjection.from_pretrained(hf_id).to(DEVICE).eval()
    feats = []
    with torch.no_grad():
        for it in manifest:
            img = Image.open(ROOT / it["frame_path"]).convert("RGB")
            x = proc(images=img, return_tensors="pt").to(DEVICE)
            out = model(**x)
            feats.append(out.image_embeds[0].cpu().numpy())
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return np.stack(feats).astype(np.float32)


def encode_vision_mae(hf_id: str, manifest: list[dict]) -> np.ndarray:
    from transformers import AutoImageProcessor, ViTModel
    proc = AutoImageProcessor.from_pretrained(hf_id)
    model = ViTModel.from_pretrained(hf_id, add_pooling_layer=False).to(DEVICE).eval()
    feats = []
    with torch.no_grad():
        for it in manifest:
            img = Image.open(ROOT / it["frame_path"]).convert("RGB")
            x = proc(images=img, return_tensors="pt").to(DEVICE)
            out = model(**x)
            feats.append(out.last_hidden_state[:, 0][0].cpu().numpy())
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return np.stack(feats).astype(np.float32)


VISION_DISPATCH = {
    "auto_pooler": encode_vision_auto_pooler,
    "clip": encode_vision_clip,
    "mae": encode_vision_mae,
}


def encode_audio_clap(hf_id: str, manifest: list[dict], sr: int = 48000) -> np.ndarray:
    """CLAP audio embedder.

    Uses `ClapAudioModelWithProjection` which returns `audio_embeds` —
    the pooled, projected, contrastively-aligned audio vector of shape
    (B, projection_dim). We avoid `ClapModel.get_audio_features` because
    in some transformers versions it returns the HTSAT backbone's spatial
    feature map (shape (1, hidden, freq, time)) instead of the projected
    embedding.
    """
    from transformers import ClapAudioModelWithProjection, ClapProcessor
    proc = ClapProcessor.from_pretrained(hf_id)
    model = ClapAudioModelWithProjection.from_pretrained(hf_id).to(DEVICE).eval()
    feats = []
    with torch.no_grad():
        for it in manifest:
            wav, sr_in = sf.read(ROOT / it["audio_path"])
            wav = _resample(wav, sr_in, sr)
            wav = _fix_audio(wav, sr, duration_s=10.0)
            # `audios=` is documented on recent transformers; some older
            # versions accept only `audio=` (singular).
            x = proc(audio=wav, sampling_rate=sr, return_tensors="pt").to(DEVICE)
            out = model(**x)
            f = out.audio_embeds  # (1, projection_dim)
            if f.ndim != 2:
                raise RuntimeError(
                    f"unexpected ClapAudioModelWithProjection output {tuple(f.shape)}"
                )
            feats.append(f[0].cpu().numpy())
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return np.stack(feats).astype(np.float32)


def encode_audio_mert(hf_id: str, manifest: list[dict], sr: int = 24000) -> np.ndarray:
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    proc = Wav2Vec2FeatureExtractor.from_pretrained(hf_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(hf_id, trust_remote_code=True).to(DEVICE).eval()
    feats = []
    with torch.no_grad():
        for it in manifest:
            wav, sr_in = sf.read(ROOT / it["audio_path"])
            wav = _resample(wav, sr_in, sr)
            wav = _fix_audio(wav, sr, duration_s=10.0)
            x = proc(wav, sampling_rate=sr, return_tensors="pt").to(DEVICE)
            out = model(**x)
            f = out.last_hidden_state.mean(dim=1)
            feats.append(f[0].cpu().numpy())
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return np.stack(feats).astype(np.float32)


AUDIO_DISPATCH = {
    "clap": encode_audio_clap,
    "mert": encode_audio_mert,
}


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings with attention mask. Matches the pooling that
    sentence-transformers/all-MiniLM-L6-v2 uses internally."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def encode_text_pool(manifest: list[dict], field: str) -> np.ndarray:
    """Encode each clip's caption pool with MiniLM (via plain HF transformers,
    avoiding the sentence-transformers / torchcodec dependency chain).

    Per clip: tokenise all captions, mean-pool token embeddings per caption,
    then average across the pool to get one vector per clip.
    """
    from transformers import AutoModel, AutoTokenizer
    hf_id = "sentence-transformers/all-MiniLM-L6-v2"
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModel.from_pretrained(hf_id).to(DEVICE).eval()
    feats = []
    with torch.no_grad():
        for it in manifest:
            caps = it[field]
            enc = tok(
                caps, padding=True, truncation=True, max_length=128,
                return_tensors="pt",
            ).to(DEVICE)
            out = model(**enc)
            per_caption = _mean_pool(out.last_hidden_state, enc["attention_mask"])
            feats.append(per_caption.mean(dim=0).cpu().numpy())
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return np.stack(feats).astype(np.float32)


def encode_and_save(
    out_path: Path,
    fn,
    *args,
    overwrite: bool = False,
    **kw,
) -> None:
    if out_path.exists() and not overwrite:
        print(f"[skip] {out_path.name} exists")
        return
    print(f"[encode] -> {out_path.name}")
    X = fn(*args, **kw)
    if X.ndim != 2:
        raise RuntimeError(f"{out_path.name} bad shape {X.shape}")
    X = l2_normalise(X)
    np.save(out_path, X)
    print(f"          shape={X.shape}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-vision", action="store_true")
    ap.add_argument("--only-audio", action="store_true")
    ap.add_argument("--only-text", action="store_true")
    ap.add_argument("--encoders", type=str, default=None,
                    help="Comma-separated names; only encode the named ones.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    EMB.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    print(f"manifest rows: {len(manifest)}")
    print(f"device: {DEVICE}")

    do_v = not (args.only_audio or args.only_text)
    do_a = not (args.only_vision or args.only_text)
    do_t = not (args.only_vision or args.only_audio)
    pick = set(args.encoders.split(",")) if args.encoders else None

    if do_v:
        for name, cfg in VISION_ENCODERS.items():
            if pick is not None and name not in pick:
                continue
            try:
                encode_and_save(
                    EMB / f"vision_{name}.npy",
                    VISION_DISPATCH[cfg["type"]],
                    cfg["hf_id"], manifest,
                    overwrite=args.overwrite,
                )
            except Exception as e:
                import traceback
                print(f"  ! vision/{name} failed: {e.__class__.__name__}: {e}")
                traceback.print_exc()

    if do_a:
        for name, cfg in AUDIO_ENCODERS.items():
            if pick is not None and name not in pick:
                continue
            try:
                extra = {k: v for k, v in cfg.items() if k not in ("hf_id", "type")}
                encode_and_save(
                    EMB / f"audio_{name}.npy",
                    AUDIO_DISPATCH[cfg["type"]],
                    cfg["hf_id"], manifest,
                    overwrite=args.overwrite,
                    **extra,
                )
            except Exception as e:
                import traceback
                print(f"  ! audio/{name} failed: {e.__class__.__name__}: {e}")
                traceback.print_exc()

    if do_t:
        try:
            encode_and_save(
                EMB / "ZV_text.npy",
                encode_text_pool, manifest, "visual_captions",
                overwrite=args.overwrite,
            )
            encode_and_save(
                EMB / "ZA_text.npy",
                encode_text_pool, manifest, "audio_captions",
                overwrite=args.overwrite,
            )
        except Exception as e:
            import traceback
            print(f"  ! text failed: {e.__class__.__name__}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
