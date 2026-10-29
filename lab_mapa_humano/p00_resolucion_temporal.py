"""Resolución temporal real de los onsets de basic-pitch.

1) Compara el mapeo fotograma→tiempo de la librería (note_creation.py:346-357) con el tiempo
   exacto que se deduce del troceado de inference.py (ventanas de 43844 muestras, salto 36164,
   15 fotogramas recortados por lado).
2) Sintetiza tonos armónicos con ataque en una muestra exacta (posiciones aleatorias, sub-fotograma),
   uno a uno, y mide el error de onset en función de la posición dentro del fotograma y de la ventana.

Uso: python p00_resolucion_temporal.py <dir_trabajo>
"""
import json
import os
import sys

import numpy as np
import soundfile as sf

import comun
from basic_pitch.constants import AUDIO_N_SAMPLES, AUDIO_SAMPLE_RATE, FFT_HOP
from basic_pitch.inference import DEFAULT_OVERLAPPING_FRAMES
import basic_pitch.note_creation as infer

AQUI = os.path.dirname(os.path.abspath(__file__))


def tiempo_exacto(j: np.ndarray) -> np.ndarray:
    olap = DEFAULT_OVERLAPPING_FRAMES * FFT_HOP
    hop = AUDIO_N_SAMPLES - olap
    por_ventana = 2 * (AUDIO_SAMPLE_RATE // FFT_HOP) - DEFAULT_OVERLAPPING_FRAMES  # 142
    w = j // por_ventana
    return (j * FFT_HOP - w * (por_ventana * FFT_HOP - hop)) / AUDIO_SAMPLE_RATE


def main(trabajo: str) -> None:
    os.makedirs(trabajo, exist_ok=True)
    res = {}
    # 1) mapeo
    j = np.arange(86 * 600)  # 10 minutos de fotogramas
    lib = infer.model_frames_to_time(len(j))
    exa = tiempo_exacto(j)
    d = (lib - exa) * 1000
    res["mapeo_lib_menos_exacto_ms"] = {
        "min": float(d.min()), "max": float(d.max()), "media": float(d.mean()),
        "fotograma_ms": 1000 * FFT_HOP / AUDIO_SAMPLE_RATE,
    }

    # 2) tonos sintéticos con onset exacto
    rng = np.random.default_rng(0)
    sr = 44100
    n = 600
    sep = 0.35
    dur_total = n * sep + 2
    y = np.zeros(int(dur_total * sr))
    ref = []
    for k in range(n):
        t0 = 1.0 + k * sep + rng.uniform(0, 0.02)
        p = int(rng.integers(45, 85))
        f0 = 440 * 2 ** ((p - 69) / 12)
        L = int(0.28 * sr)
        tt = np.arange(L) / sr
        env = np.exp(-tt / 0.25)
        env[: int(0.001 * sr)] *= np.linspace(0, 1, int(0.001 * sr))
        s = sum((1 / h) * np.sin(2 * np.pi * h * f0 * tt) for h in range(1, 7) if h * f0 < sr / 2)
        i0 = int(round(t0 * sr))
        y[i0 : i0 + L] += 0.3 * env * s
        ref.append((i0 / sr, i0 / sr + 0.28, p, 0.8))
    wav = os.path.join(trabajo, "tonos.wav")
    sf.write(wav, y, sr)
    out = comun.salida_modelo(wav)
    est = comun.notas(out)
    ev = comun.evaluar(ref, est)
    res["tonos_sinteticos"] = ev

    # error frente a la posición del onset dentro del fotograma y dentro de la ventana
    import mir_eval, pretty_midi

    ri = np.array([[s, e] for s, e, _, _ in ref]); rp = np.array([pretty_midi.note_number_to_hz(p) for *_, p, _ in ref])
    ei = np.array([[s, e] for s, e, _, _ in est]); ep = np.array([pretty_midi.note_number_to_hz(p) for *_, p, _ in est])
    pares = mir_eval.transcription.match_notes(ri, rp, ei, ep, onset_tolerance=0.05, offset_ratio=None)
    err = np.array([ei[b, 0] - ri[a, 0] for a, b in pares]) * 1000
    t_ref = np.array([ri[a, 0] for a, _ in pares])
    hop_s = (AUDIO_N_SAMPLES - DEFAULT_OVERLAPPING_FRAMES * FFT_HOP) / AUDIO_SAMPLE_RATE
    pos_ventana = (t_ref % hop_s) / hop_s
    tercios = {}
    for a, b in [(0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1)]:
        m = (pos_ventana >= a) & (pos_ventana < b)
        tercios[f"{a:.2f}-{b:.2f}"] = {"n": int(m.sum()), "mediana_ms": float(np.median(err[m]))}
    res["tonos_error_por_posicion_en_ventana"] = tercios
    valores, cuentas = np.unique(np.round(err, 1), return_counts=True)
    res["tonos_desv_tipica_ms"] = float(np.std(err))
    res["tonos_rango_ms"] = [float(err.min()), float(err.max())]

    # 3) mismo audio, sustituyendo el mapeo de la librería por el exacto
    original = infer.model_frames_to_time
    infer.model_frames_to_time = lambda n: tiempo_exacto(np.arange(n))
    try:
        res["tonos_con_mapeo_exacto"] = comun.evaluar(ref, comun.notas(out))
    finally:
        infer.model_frames_to_time = original
    json.dump(res, open(os.path.join(AQUI, "resultados", "p00_resolucion.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
