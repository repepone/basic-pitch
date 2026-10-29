"""Complemento de p02: mismas salidas cacheadas, tolerancia de onset de 100 ms en vez de 50 ms.

En viento y cuerda sintetizados el ataque de la muestra es lento y basic-pitch da el onset 15-30 ms
tarde respecto al note-on del MIDI; con ±50 ms parte de esas notas cuentan como perdida + inventada.
Uso: python p02b_tolerancia.py <dir_trabajo>
"""
import os
import sys

import numpy as np
import pandas as pd

import comun
import p02_onsets_sintetizado as p02

AQUI = os.path.dirname(os.path.abspath(__file__))


def main(trabajo: str) -> None:
    filas = []
    for familia in p02.CONFIG:
        tr, te, t0 = [], [], 0.0
        for k, coral in enumerate(p02.CORALES):
            base = os.path.join(trabajo, "sint", f"{familia.split(':')[-1].strip().split(' ')[0]}_{len(p02.CONFIG[familia][0])}_{k}")
            ref = p02.construir_midi(coral, familia, base + "_p02b.mid", np.random.default_rng(k))
            est = comun.notas(comun.salida_modelo(None, base + "_seco_bp.npz"))
            tr += [(s + t0, e + t0, p, v) for s, e, p, v in ref]
            te += [(s + t0, e + t0, p, v) for s, e, p, v in est]
            t0 += 10000
        for tol in (0.05, 0.1):
            filas.append({"familia": familia, "reverb": "no", "tolerancia_ms": int(tol * 1000), **comun.evaluar(tr, te, tol)})
        print(filas[-1], flush=True)
    pd.DataFrame(filas).to_csv(os.path.join(AQUI, "resultados", "p02b_tolerancia.csv"), index=False, float_format="%.2f")


if __name__ == "__main__":
    main(sys.argv[1])
