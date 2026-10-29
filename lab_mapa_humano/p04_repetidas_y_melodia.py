"""Dos comprobaciones de la lógica de note_creation.py:

1) Notas repetidas del mismo tono: el bucle de note_creation.py:416-436 corta una nota donde empieza la
   siguiente del mismo tono (energía ya puesta a 0) y luego descarta las de <= min_note_len fotogramas.
   Se sintetizan (piano FluidR3) trenes de 24 notas repetidas con distintos IOI y se cuenta cuántas salen.
2) Efecto de melodia_trick (notas sin onset detectado, note_creation.py:449-509) sobre piano real.

Uso: python p04_repetidas_y_melodia.py <dir_asap> <dir_maestro_sel> <dir_trabajo>
"""
import csv
import json
import os
import sys

import pandas as pd
import pretty_midi

import comun
from p01_onsets_piano_maestro import tramo

AQUI = os.path.dirname(os.path.abspath(__file__))
IOIS_MS = [60, 80, 100, 120, 140, 160, 200]


def main(asap: str, sel_dir: str, trabajo: str) -> None:
    res = {"repetidas": [], "melodia_trick_piano_maestro": {}}
    d = os.path.join(trabajo, "repetidas")
    os.makedirs(d, exist_ok=True)
    for programa, nombre in [(0, "piano"), (40, "violín")]:
        for ioi in IOIS_MS:
            pm = pretty_midi.PrettyMIDI()
            ins = pretty_midi.Instrument(program=programa)
            ref = []
            t = 1.0
            for tono in (55, 64, 72):
                for _ in range(24):
                    ins.notes.append(pretty_midi.Note(velocity=90, pitch=tono, start=t, end=t + 0.8 * ioi / 1000))
                    ref.append((t, t + 0.8 * ioi / 1000, tono, 0.7))
                    t += ioi / 1000
                t += 1.0
            pm.instruments.append(ins)
            mid = os.path.join(d, f"{nombre}_{ioi}.mid")
            wav = mid.replace(".mid", ".wav")
            pm.write(mid)
            comun.render_fluidsynth(mid, wav)
            out = comun.salida_modelo(wav)
            for ml in (127.7, 58.0):
                e = comun.evaluar(ref, comun.notas(out, minimum_note_length_ms=ml))
                res["repetidas"].append({"instrumento": nombre, "ioi_ms": ioi, "minimum_note_length_ms": ml,
                                         "n_ref": e["n_ref"], "casadas": e["casadas"], "pct_perdidas": round(e["pct_perdidas"], 1)})
                print(res["repetidas"][-1], flush=True)
    # melodia trick sobre las salidas cacheadas de p01
    sel = list(csv.DictReader(open(os.path.join(AQUI, "seleccion_asap.csv"))))
    meta = pd.read_csv(os.path.join(asap, "metadata.csv"))
    for mt in (True, False):
        tr, te, t0 = [], [], 0.0
        for r in sel:
            cache = os.path.join(trabajo, f"{r['id']}_tramo_bp.npz")
            ini, fin = tramo(meta, r["folder"], r["performance"])
            ref = comun.notas_midi(os.path.join(sel_dir, f"{r['id']}.midi"), ini, fin)
            est = comun.notas(comun.salida_modelo(None, cache), melodia_trick=mt)
            tr += [(s + t0, e + t0, p, v) for s, e, p, v in ref]
            te += [(s + t0, e + t0, p, v) for s, e, p, v in est]
            t0 += 10000
        res["melodia_trick_piano_maestro"]["si" if mt else "no"] = comun.evaluar(tr, te)
    json.dump(res, open(os.path.join(AQUI, "resultados", "p04_repetidas_y_melodia.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res["melodia_trick_piano_maestro"], indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
