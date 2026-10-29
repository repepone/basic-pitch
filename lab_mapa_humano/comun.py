"""Utilidades compartidas del laboratorio «mapa humano».

Todo corre con el modelo ONNX que trae el repo (basic_pitch/saved_models/icassp_2022/nmp.onnx),
porque TensorFlow no está instalado en este entorno.
"""
import os
import subprocess
from typing import Dict, List, Optional, Tuple

import mir_eval
import numpy as np
import pretty_midi

from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
from basic_pitch.inference import Model, run_inference
import basic_pitch.note_creation as infer
from basic_pitch.constants import AUDIO_SAMPLE_RATE, FFT_HOP

SF2 = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
_MODELO: Optional[Model] = None


def modelo() -> Model:
    global _MODELO
    if _MODELO is None:
        _MODELO = Model(build_icassp_2022_model_path(FilenameSuffix.onnx))
    return _MODELO


def salida_modelo(audio_path: str, cache: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Salida cruda del modelo (note/onset/contour), cacheada en .npz para barrer umbrales sin re-inferir."""
    if cache and os.path.exists(cache):
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    out = run_inference(audio_path, modelo())
    if cache:
        tmp = cache + ".tmp.npz"
        np.savez_compressed(tmp, **out)
        os.replace(tmp, cache)  # escritura atómica: un proceso cortado no deja una caché rota
    return out


def notas(
    out: Dict[str, np.ndarray],
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length_ms: float = 127.7,
    melodia_trick: bool = True,
) -> List[Tuple[float, float, int, float]]:
    """Igual que basic_pitch.inference.predict (inference.py:469-480) pero sobre la salida cacheada."""
    min_note_len = int(np.round(minimum_note_length_ms / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))
    copia = {k: v.copy() for k, v in out.items()}
    _, ev = infer.model_output_to_notes(
        copia,
        onset_thresh=onset_threshold,
        frame_thresh=frame_threshold,
        min_note_len=min_note_len,
        melodia_trick=melodia_trick,
        include_pitch_bends=False,
    )
    return sorted((float(s), float(e), int(p), float(a)) for s, e, p, a, _ in ev)


def notas_midi(path: str, desde: float = 0.0, hasta: Optional[float] = None) -> List[Tuple[float, float, int, float]]:
    pm = pretty_midi.PrettyMIDI(path)
    res = []
    for ins in pm.instruments:
        if ins.is_drum:
            continue
        for n in ins.notes:
            if n.start < desde or (hasta is not None and n.start >= hasta):
                continue
            res.append((n.start - desde, n.end - desde, n.pitch, n.velocity / 127))
    return sorted(res)


def evaluar(ref, est, tol: float = 0.05) -> Dict[str, float]:
    """Casa notas por (tono exacto ±50 cents, onset ±tol) con mir_eval y resume el error de onset."""
    ri = np.array([[s, max(e, s + 1e-3)] for s, e, _, _ in ref]).reshape(-1, 2)
    rp = np.array([pretty_midi.note_number_to_hz(p) for _, _, p, _ in ref])
    ei = np.array([[s, max(e, s + 1e-3)] for s, e, _, _ in est]).reshape(-1, 2)
    ep = np.array([pretty_midi.note_number_to_hz(p) for _, _, p, _ in est])
    pares = mir_eval.transcription.match_notes(ri, rp, ei, ep, onset_tolerance=tol, pitch_tolerance=50.0, offset_ratio=None)
    err = np.array([ei[j, 0] - ri[i, 0] for i, j in pares]) * 1000
    n_ref, n_est, m = len(ref), len(est), len(pares)
    r = {
        "n_ref": n_ref,
        "n_est": n_est,
        "casadas": m,
        "perdidas": n_ref - m,
        "inventadas": n_est - m,
        "pct_perdidas": 100 * (n_ref - m) / max(n_ref, 1),
        "pct_inventadas": 100 * (n_est - m) / max(n_est, 1),
    }
    if m:
        r.update(
            {
                "mediana_ms": float(np.median(err)),
                "p25_ms": float(np.percentile(err, 25)),
                "p75_ms": float(np.percentile(err, 75)),
                "mediana_abs_ms": float(np.median(np.abs(err))),
                "p90_abs_ms": float(np.percentile(np.abs(err), 90)),
                "pct_abs_le_20ms": float(100 * np.mean(np.abs(err) <= 20)),
            }
        )
    return r


def render_fluidsynth(midi_path: str, wav_path: str, reverb: bool = False, sr: int = 44100) -> None:
    """Sintetiza un MIDI con FluidR3_GM (MIT). Con reverb=False se apagan reverb y chorus."""
    cmd = ["fluidsynth", "-ni", "-q", "-g", "0.6", "-r", str(sr), "-F", wav_path]
    if not reverb:
        cmd += ["-R", "0", "-C", "0"]
    else:
        cmd += ["-R", "1", "-C", "0", "-o", "synth.reverb.room-size=0.9", "-o", "synth.reverb.level=0.9",
                "-o", "synth.reverb.damp=0.2", "-o", "synth.reverb.width=1.0"]
    cmd += [SF2, midi_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
