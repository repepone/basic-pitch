"""Mapa «nota de la partitura → ms» fuera del piano, con audio sintetizado (verdad exacta).

Corales de Bach (music21, dominio público) → MusicXML (partitura) y → MIDI humanizado (como p02) →
FluidR3 (cuarteto de cuerda, conjunto mixto, piano; con y sin reverberación). Métodos A, B, C de p03 y E
(= receta_mapa.py) de p05.
La verdad de cada nota de la partitura es el tiempo con que se escribió en el MIDI.

Uso: python p07_mapa_no_piano.py <dir_trabajo>
"""
import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd
from music21 import corpus

import comun
import p02_onsets_sintetizado as p02
import p03_mapa_partitura as p03

AQUI = os.path.dirname(os.path.abspath(__file__))
FAMILIAS = ["cuerda: cuarteto", "conjunto: mixto (vl, cl, trompa, vc)", "piano (sintetizado)"]


def main(trabajo: str) -> None:
    import partitura as pt

    d = os.path.join(trabajo, "p07")
    os.makedirs(d, exist_ok=True)
    filas = []
    for familia in FAMILIAS:
        for reverb in (False, True):
            errs = {m: [] for m in "ABCE"}
            marcas = []
            for k, coral in enumerate(p02.CORALES):
                xml = os.path.join(d, f"coral{k}.musicxml")
                if not os.path.exists(xml):
                    corpus.parse(coral).write("musicxml", fp=xml)
                rng = np.random.default_rng(k)
                base = os.path.join(d, f"{familia.split(':')[0].strip().split(' ')[0]}_{k}")
                claves = []
                p02.construir_midi(coral, familia, base + ".mid", rng, claves)
                wav = base + ("_rev" if reverb else "_seco") + ".wav"
                if not os.path.exists(wav):
                    comun.render_fluidsynth(base + ".mid", wav, reverb=reverb)
                # verdad por (onset en negras, tono): el MIDI se construyó desde las mismas voces de music21
                clave_t = {}
                for off, p, s in claves:
                    clave_t.setdefault((off, p), []).append(s)
                with contextlib.redirect_stdout(io.StringIO()):
                    score = pt.load_musicxml(xml)
                    part = pt.score.merge_parts(score.parts[:4])
                    # el MusicXML de music21 no trae id en <note>: partitura los deja como None; se asignan
                    for j, n in enumerate(part.notes_tied):
                        n.id = f"n{j}"
                    na = part.note_array(include_grace_notes=True)
                # sólo las voces que suenan en esta familia, con su transposición
                info = pd.DataFrame({"id": na["id"], "pitch": na["pitch"], "onset_beat": na["onset_beat"],
                                     "onset_q": na["onset_quarter"], "is_grace": na["is_grace"].astype(bool)})
                # music21 y partitura cuentan la anacrusa de forma distinta: se busca el corrimiento en negras
                cands = {round(o - q, 4) for (o, _) in list(clave_t)[:8] for q in info.onset_q[:8]}
                desp = max(sorted(cands), key=lambda c: sum((round(q + c, 4), p) in clave_t for q, p in zip(info.onset_q, info.pitch)))
                info["verdad"] = [np.mean(clave_t[(round(q + desp, 4), p)]) if (round(q + desp, 4), p) in clave_t else np.nan
                                  for q, p in zip(info.onset_q, info.pitch)]
                info = info.dropna(subset=["verdad"]).reset_index(drop=True)
                info["adorno"] = False
                na2 = na[np.isin(na["id"], info.id.values)]
                order = {i: j for j, i in enumerate(na2["id"])}
                info = info.iloc[np.argsort([order[i] for i in info.id])].reset_index(drop=True)
                bp = comun.notas(comun.salida_modelo(wav, wav.replace(".wav", "_bp.npz")))
                tA, dA = p03.metodo_simbolico(na2, info, bp, part)
                tB = p03.metodo_audio(wav, na2, info)
                tC, dC = p03.metodo_hibrido(tB, info, bp)
                tE = np.where(dA, tA, tC)  # método de receta_mapa.py
                for m, t in zip("ABCE", (tA, tB, tC, tE)):
                    errs[m].append((t - info.verdad.values) * 1000)
                marcas.append(np.abs(tA - tB) > 0.060)
            mk = np.concatenate(marcas)
            eE = np.abs(np.concatenate(errs["E"]))
            for m in "ABCE":
                extra = {"pct_marcadas": 100 * mk.mean(), "pct_gt100_no_marcadas": 100 * np.mean(eE[~mk] > 100)} if m == "E" else {}
                filas.append({"familia": familia, "reverb": "sí" if reverb else "no", "metodo": m,
                              **p03.resumir(np.concatenate(errs[m])), **extra})
                print(filas[-1], flush=True)
    pd.DataFrame(filas).to_csv(os.path.join(AQUI, "resultados", "p07_mapa_no_piano.csv"), index=False, float_format="%.2f")


if __name__ == "__main__":
    main(sys.argv[1])
