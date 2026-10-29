"""Precisión de onsets de basic-pitch sobre piano real (MAESTRO v3, MIDI del Disklavier como verdad).

Se evalúa el tramo de cada grabación que ASAP asocia a la obra (columnas start/end de metadata.csv).
Uso: python p01_onsets_piano_maestro.py <dir_asap> <dir_maestro_sel> <dir_trabajo>
"""
import csv
import itertools
import json
import os
import sys

import pandas as pd
import soundfile as sf

import comun

AQUI = os.path.dirname(os.path.abspath(__file__))
UMBRALES_ONSET = [0.3, 0.4, 0.5, 0.6, 0.7]
UMBRALES_FRAME = [0.2, 0.3, 0.4]
LONG_MIN = [58.0, 127.7]


def tramo(meta: pd.DataFrame, folder: str, perf: str):
    f = meta[meta.midi_performance == f"{folder}/{perf}.mid"].iloc[0]
    ini = 0.0 if pd.isna(f.start) else float(f.start)
    fin = None if pd.isna(f.end) else float(f.end)
    return ini, fin


def preparar(asap: str, sel_dir: str, trabajo: str):
    meta = pd.read_csv(os.path.join(asap, "metadata.csv"))
    piezas = []
    for r in csv.DictReader(open(os.path.join(AQUI, "seleccion_asap.csv"))):
        ini, fin = tramo(meta, r["folder"], r["performance"])
        wav = os.path.join(trabajo, f"{r['id']}_tramo.wav")
        if not os.path.exists(wav):
            info = sf.info(os.path.join(sel_dir, f"{r['id']}.wav"))
            a = int(ini * info.samplerate)
            b = None if fin is None else int(fin * info.samplerate)
            y, sr = sf.read(os.path.join(sel_dir, f"{r['id']}.wav"), start=a, stop=b)
            sf.write(wav, y, sr)
        ref = comun.notas_midi(os.path.join(sel_dir, f"{r['id']}.midi"), ini, fin)
        out = comun.salida_modelo(wav, os.path.join(trabajo, f"{r['id']}_tramo_bp.npz"))
        piezas.append((r, ref, out, ini, fin))
    return piezas


def main(asap: str, sel_dir: str, trabajo: str) -> None:
    os.makedirs(trabajo, exist_ok=True)
    piezas = preparar(asap, sel_dir, trabajo)
    filas = []
    for r, ref, out, ini, fin in piezas:
        e = comun.evaluar(ref, comun.notas(out))
        filas.append({"id": r["id"], "rasgo": r["familia_rasgo"], "seg_s": round((fin or 0) - ini, 1) if fin else None, **e})
    df = pd.DataFrame(filas)
    # agregado: juntando todos los errores
    todos_ref, todos_est, t0 = [], [], 0.0
    for r, ref, out, ini, fin in piezas:
        est = comun.notas(out)
        todos_ref += [(s + t0, e + t0, p, v) for s, e, p, v in ref]
        todos_est += [(s + t0, e + t0, p, v) for s, e, p, v in est]
        t0 += 10000
    agg = comun.evaluar(todos_ref, todos_est)
    df.loc[len(df)] = {"id": "TOTAL", "rasgo": "", **agg}
    df.to_csv(os.path.join(AQUI, "resultados", "p01_piano_por_pieza.csv"), index=False, float_format="%.2f")
    print(df.to_string())

    # barrido de umbrales (sobre el total)
    barrido = []
    for ot, ft, ml in itertools.product(UMBRALES_ONSET, UMBRALES_FRAME, LONG_MIN):
        tr, te, t0 = [], [], 0.0
        for r, ref, out, ini, fin in piezas:
            est = comun.notas(out, ot, ft, ml)
            tr += [(s + t0, e + t0, p, v) for s, e, p, v in ref]
            te += [(s + t0, e + t0, p, v) for s, e, p, v in est]
            t0 += 10000
        barrido.append({"onset_threshold": ot, "frame_threshold": ft, "minimum_note_length_ms": ml, **comun.evaluar(tr, te)})
    db = pd.DataFrame(barrido)
    db.to_csv(os.path.join(AQUI, "resultados", "p01_piano_barrido_umbrales.csv"), index=False, float_format="%.2f")
    print(db[["onset_threshold", "frame_threshold", "minimum_note_length_ms", "n_est", "pct_perdidas", "pct_inventadas", "mediana_ms", "p25_ms", "p75_ms", "p90_abs_ms"]].to_string())

    # tolerancia mayor: ¿cuántas perdidas lo son por >50 ms y no por ausencia?
    tr, te, t0 = [], [], 0.0
    for r, ref, out, ini, fin in piezas:
        tr += [(s + t0, e + t0, p, v) for s, e, p, v in ref]
        te += [(s + t0, e + t0, p, v) for s, e, p, v in comun.notas(out)]
        t0 += 10000
    json.dump({"tol_100ms": comun.evaluar(tr, te, 0.1), "tol_50ms": comun.evaluar(tr, te, 0.05), "tol_25ms": comun.evaluar(tr, te, 0.025)},
              open(os.path.join(AQUI, "resultados", "p01_piano_tolerancias.json"), "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
