"""¿Qué son las notas «inventadas» y «perdidas»? (sobre las salidas cacheadas de p01 y p02)

Inventada «armónica»: hay una nota real sonando (onset ±50 ms o sostenida) a -12, +12, +19, +24, +28 o
+31 semitonos. Inventada «repetición»: misma altura que una nota real que sigue sonando (re-ataque falso).
Perdida «en acorde»: la nota real comparte onset (±30 ms) con otra nota real.

Uso: python p08_tipos_de_error.py <dir_asap> <dir_maestro_sel> <dir_trabajo>
"""
import csv
import os
import sys

import mir_eval
import numpy as np
import pandas as pd
import pretty_midi

import comun
from p01_onsets_piano_maestro import tramo

AQUI = os.path.dirname(os.path.abspath(__file__))
ARMONICOS = {-12, 12, 19, 24, 28, 31}


def clasificar(ref, est, tol=0.05):
    ri = np.array([[s, max(e, s + 1e-3)] for s, e, _, _ in ref]); rp = np.array([p for *_, p, _ in ref])
    ei = np.array([[s, max(e, s + 1e-3)] for s, e, _, _ in est]); ep = np.array([p for *_, p, _ in est])
    pares = mir_eval.transcription.match_notes(ri, pretty_midi.note_number_to_hz(rp), ei, pretty_midi.note_number_to_hz(ep),
                                               onset_tolerance=tol, offset_ratio=None)
    casadas_r = {i for i, _ in pares}
    casadas_e = {j for _, j in pares}
    inv = {"armonico": 0, "re-ataque misma nota": 0, "otro": 0}
    for j in set(range(len(est))) - casadas_e:
        t, p = ei[j, 0], ep[j]
        sonando = (ri[:, 0] <= t + tol) & (ri[:, 1] >= t - tol)
        if np.any(sonando & np.isin(p - rp, list(ARMONICOS))):
            inv["armonico"] += 1
        elif np.any(sonando & (rp == p)):
            inv["re-ataque misma nota"] += 1
        else:
            inv["otro"] += 1
    perd = {"en acorde": 0, "nota sola": 0}
    for i in set(range(len(ref))) - casadas_r:
        juntas = np.sum(np.abs(ri[:, 0] - ri[i, 0]) <= 0.03) > 1
        perd["en acorde" if juntas else "nota sola"] += 1
    return len(ref), len(est), inv, perd


def main(asap, sel_dir, trabajo):
    filas = []
    meta = pd.read_csv(os.path.join(asap, "metadata.csv"))
    for r in csv.DictReader(open(os.path.join(AQUI, "seleccion_asap.csv"))):
        ini, fin = tramo(meta, r["folder"], r["performance"])
        ref = comun.notas_midi(os.path.join(sel_dir, f"{r['id']}.midi"), ini, fin)
        est = comun.notas(comun.salida_modelo(None, os.path.join(trabajo, f"{r['id']}_tramo_bp.npz")))
        n_r, n_e, inv, perd = clasificar(ref, est)
        filas.append({"conjunto": "piano real (MAESTRO)", "pieza": r["id"], "n_ref": n_r, "n_est": n_e,
                      **{f"inv_{k}": v for k, v in inv.items()}, **{f"perd_{k}": v for k, v in perd.items()}})
    df = pd.DataFrame(filas)
    total = df.drop(columns=["conjunto", "pieza"]).sum()
    df.loc[len(df)] = {"conjunto": "piano real (MAESTRO)", "pieza": "TOTAL", **total.to_dict()}

    # familias sintetizadas de p02 (sin reverberación), sumando los 10 corales
    import p02_onsets_sintetizado as p02

    for familia in p02.CONFIG:
        acum = None
        for k, coral in enumerate(p02.CORALES):
            base = os.path.join(trabajo, "sint", f"{familia.split(':')[-1].strip().split(' ')[0]}_{len(p02.CONFIG[familia][0])}_{k}")
            ref = p02.construir_midi(coral, familia, base + "_p08.mid", np.random.default_rng(k))
            est = comun.notas(comun.salida_modelo(None, base + "_seco_bp.npz"))
            n_r, n_e, inv, perd = clasificar(ref, est)
            fila = {"n_ref": n_r, "n_est": n_e, **{f"inv_{a}": b for a, b in inv.items()}, **{f"perd_{a}": b for a, b in perd.items()}}
            acum = fila if acum is None else {a: acum[a] + fila[a] for a in acum}
        df.loc[len(df)] = {"conjunto": "sintetizado, seco", "pieza": familia, **acum}
    df.to_csv(os.path.join(AQUI, "resultados", "p08_tipos_de_error.csv"), index=False)
    print(df.to_string())


if __name__ == "__main__":
    main(*sys.argv[1:4])
