"""Efecto de la reverberación sobre onsets y sobre el mapa de partitura.

Se convoluciona el audio de MAESTRO (sala de concurso, poca reverberación añadida) con una respuesta al
impulso sintética: ruido blanco con caída exponencial (RT60 dado), mezcla 50 % seco / 50 % húmedo.
Después se repiten las mediciones de p01 (onsets) y p03/p05 (métodos A, B y E = receta) sobre el audio reverberado.

Uso: python p06_reverberacion.py <dir_asap> <dir_maestro_sel> <dir_trabajo> [ids...]
"""
import csv
import os
import sys

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import fftconvolve

import comun
import p03_mapa_partitura as p03
from p01_onsets_piano_maestro import tramo

AQUI = os.path.dirname(os.path.abspath(__file__))
RT60S = [0.0, 1.0, 2.5]
POR_DEFECTO = ["schubert_mm3", "mozart_12_1", "haydn_31_1"]


def reverberar(y: np.ndarray, sr: int, rt60: float, semilla: int = 0) -> np.ndarray:
    if rt60 <= 0:
        return y
    rng = np.random.default_rng(semilla)
    n = int(rt60 * 1.2 * sr)
    t = np.arange(n) / sr
    ir = rng.standard_normal(n) * np.exp(-6.91 * t / rt60)  # -60 dB en rt60
    ir /= np.sqrt(np.sum(ir ** 2))
    if y.ndim == 1:
        y = y[:, None]
    humedo = np.stack([fftconvolve(y[:, c], ir)[: len(y)] for c in range(y.shape[1])], axis=1)
    mezcla = 0.5 * y + 0.5 * humedo
    return mezcla / max(1.0, np.max(np.abs(mezcla)) / 0.99)


def main(asap, sel_dir, trabajo, ids=None):
    ids = ids or POR_DEFECTO
    meta = pd.read_csv(os.path.join(asap, "metadata.csv"))
    sel = [r for r in csv.DictReader(open(os.path.join(AQUI, "seleccion_asap.csv"))) if r["id"] in ids]
    filas = []
    for r in sel:
        carpeta = os.path.join(asap, r["folder"])
        tsv = os.path.join(carpeta, f"{r['performance']}_note_alignments", "note_alignment.tsv")
        na, info, unf = p03.cargar_partitura(os.path.join(carpeta, "xml_score.musicxml"), p03.ids_tsv(tsv))
        ini, fin = tramo(meta, r["folder"], r["performance"])
        ref = comun.notas_midi(os.path.join(sel_dir, f"{r['id']}.midi"), ini, fin)
        ref_asap = comun.notas_midi(os.path.join(carpeta, f"{r['performance']}.mid"))
        off = p03.desfase(ref, ref_asap)
        V = {k: v + off for k, v in p03.verdad(tsv).items()}
        verdad_v = np.array([V.get(i, np.nan) for i in info.id])
        y, sr = sf.read(os.path.join(trabajo, f"{r['id']}_tramo.wav"))
        for rt in RT60S:
            wav = os.path.join(trabajo, f"{r['id']}_tramo_rt{rt:.1f}.wav")
            if not os.path.exists(wav):
                sf.write(wav, reverberar(y, sr, rt), sr)
            bp = comun.notas(comun.salida_modelo(wav, wav.replace(".wav", "_bp.npz")))
            ev = comun.evaluar(ref, bp)
            tA, dA = p03.metodo_simbolico(na, info, bp, unf)
            tB = p03.metodo_audio(wav, na, info)
            tC, dC = p03.metodo_hibrido(tB, info, bp)
            tE = np.where(dA, tA, tC)  # método de receta_mapa.py
            marcadas = np.abs(tA - tB) > 0.060
            fila = {"pieza": r["id"], "rt60_s": rt, "onsets_pct_perdidas": ev["pct_perdidas"],
                    "onsets_pct_inventadas": ev["pct_inventadas"], "onsets_mediana_ms": ev.get("mediana_ms"),
                    "onsets_p90_abs_ms": ev.get("p90_abs_ms")}
            errE = np.abs(tE - verdad_v) * 1000
            fila["E_pct_marcadas"] = 100 * np.mean(marcadas[~np.isnan(errE)])
            fila["E_pct_gt100_no_marcadas"] = 100 * np.mean((errE > 100)[~marcadas & ~np.isnan(errE)])
            for nombre, t in [("A", tA), ("B", tB), ("E", tE)]:
                s = p03.resumir((t - verdad_v) * 1000)
                fila[f"{nombre}_mediana_abs_ms"] = s["mediana_abs_ms"]
                fila[f"{nombre}_pct_le_50ms"] = s["pct_le_50ms"]
                fila[f"{nombre}_pct_gt_200ms"] = s["pct_gt_200ms"]
            filas.append(fila)
            print(fila, flush=True)
    pd.DataFrame(filas).to_csv(os.path.join(AQUI, "resultados", "p06_reverberacion.csv"), index=False, float_format="%.2f")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:] or None)
