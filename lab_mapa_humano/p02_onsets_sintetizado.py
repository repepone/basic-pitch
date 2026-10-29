"""Precisión de onsets por familia de instrumento, con audio sintetizado desde MIDI (verdad exacta).

Fuente musical: corales de Bach del corpus de music21 (dominio público). Se «humaniza» el tempo
(rubato sinusoidal ±15 %, jitter de ±10 ms) y se renderiza con FluidR3_GM (licencia MIT) seco y con
reverberación fuerte. Ojo: sintetizado ≠ grabación humana; ver informe.

Uso: python p02_onsets_sintetizado.py <dir_trabajo>
"""
import os
import sys

import numpy as np
import pandas as pd
import pretty_midi
from music21 import corpus

import comun

AQUI = os.path.dirname(os.path.abspath(__file__))
CORALES = ["bach/bwv269", "bach/bwv347", "bach/bwv153.1", "bach/bwv86.6", "bach/bwv267", "bach/bwv281",
           "bach/bwv17.7", "bach/bwv40.8", "bach/bwv248.12-2", "bach/bwv38.6"]

# familia -> (programas GM por voz S,A,T,B ; voces usadas ; transposición en semitonos por voz)
CONFIG = {
    "piano (sintetizado)": ([0, 0, 0, 0], [0, 1, 2, 3], [0, 0, 0, 0]),
    "solo: flauta": ([73], [0], [12]),
    "solo: clarinete": ([71], [0], [0]),
    "solo: violín": ([40], [0], [0]),
    "solo: violonchelo": ([42], [3], [0]),
    "solo: trompeta": ([56], [0], [0]),
    "cuerda: cuarteto": ([40, 40, 41, 42], [0, 1, 2, 3], [0, 0, 0, 0]),
    "cuerda: orquesta (String Ens.)": ([48, 48, 48, 48], [0, 1, 2, 3], [0, 0, 0, 0]),
    "conjunto: maderas": ([73, 68, 71, 70], [0, 1, 2, 3], [12, 0, 0, 0]),
    "conjunto: mixto (vl, cl, trompa, vc)": ([40, 71, 60, 42], [0, 1, 2, 3], [0, 0, 0, 0]),
}


def voces(coral: str):
    s = corpus.parse(coral)
    res = []
    for parte in s.parts[:4]:
        notas = []
        for n in parte.flatten().notes:
            if n.isChord:
                n = n.notes[-1]
            notas.append((float(n.offset), float(n.quarterLength), int(n.pitch.midi)))
        res.append(notas)
    return res


def humanizar(t_negras: float, rng, bpm=72.0):
    # tempo con rubato suave: integra 60/bpm*(1+0.15 sin)
    return (60 / bpm) * (t_negras + 0.15 * 4 / (2 * np.pi) * (1 - np.cos(2 * np.pi * t_negras / 4)))


def construir_midi(coral: str, familia: str, ruta: str, rng, claves: list = None) -> list:
    """Escribe el MIDI humanizado y devuelve las notas (verdad). Si se pasa `claves`, añade a esa lista
    (onset en negras, tono sonoro, tiempo en s) por nota, para casarlas con la partitura (p07)."""
    progs, usadas, transp = CONFIG[familia]
    vs = voces(coral)
    pm = pretty_midi.PrettyMIDI()
    ref = []
    for prog, iv, tr in zip(progs, usadas, transp):
        ins = pretty_midi.Instrument(program=prog)
        for off, ql, p in vs[iv]:
            s = 1.0 + humanizar(off, rng) + rng.uniform(-0.01, 0.01)
            e = 1.0 + humanizar(off + ql, rng) - 0.03
            ins.notes.append(pretty_midi.Note(velocity=int(rng.integers(70, 105)), pitch=p + tr, start=s, end=max(e, s + 0.08)))
            ref.append((s, max(e, s + 0.08), p + tr, 0.8))
            if claves is not None:
                claves.append((round(off, 4), p + tr, s))
        pm.instruments.append(ins)
    pm.write(ruta)
    return sorted(ref)


def main(trabajo: str) -> None:
    trabajo = os.path.join(trabajo, "sint")
    os.makedirs(trabajo, exist_ok=True)
    filas = []
    for familia in CONFIG:
        for reverb in (False, True):
            tr, te, t0 = [], [], 0.0
            for k, coral in enumerate(CORALES):
                rng = np.random.default_rng(k)
                base = os.path.join(trabajo, f"{familia.split(':')[-1].strip().split(' ')[0]}_{len(CONFIG[familia][0])}_{k}")
                mid = base + ".mid"
                ref = construir_midi(coral, familia, mid, rng)
                wav = base + ("_rev" if reverb else "_seco") + ".wav"
                if not os.path.exists(wav):
                    comun.render_fluidsynth(mid, wav, reverb=reverb)
                out = comun.salida_modelo(wav, wav.replace(".wav", "_bp.npz"))
                est = comun.notas(out)
                tr += [(s + t0, e + t0, p, v) for s, e, p, v in ref]
                te += [(s + t0, e + t0, p, v) for s, e, p, v in est]
                t0 += 10000
            filas.append({"familia": familia, "reverb": "sí" if reverb else "no", **comun.evaluar(tr, te)})
            print(filas[-1], flush=True)
    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(AQUI, "resultados", "p02_sintetizado_por_familia.csv"), index=False, float_format="%.2f")
    print(df.to_string())


if __name__ == "__main__":
    main(sys.argv[1])
