"""Combina los métodos de p03 nota a nota y mide la marca «revisar» de la receta.

Lee <dir_trabajo>/por_nota_<id>.csv (escritos por p03) y compara:
  A, B, C, T  (como en p03)
  D consenso : A si A tiene nota casada y |A-B| <= 150 ms; si no, C.
  E A+C      : A si A tiene nota casada; si no, C (en vez de interpolar).
Para E (el método de receta_mapa.py) mide la marca «revisar» de la receta: |A-B| > 60 ms. Cuenta qué fracción de notas se marca y cuántos errores gruesos (>100 ms) atrapa.

Uso: python p05_combinaciones.py <dir_trabajo>
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

from p03_mapa_partitura import resumir

AQUI = os.path.dirname(os.path.abspath(__file__))


def combinar(df: pd.DataFrame, nombre: str):
    if nombre == "D consenso":
        ok = df.d_A & ((df.t_A - df.t_B).abs() <= 0.150)
        return np.where(ok, df.t_A, df.t_C), (ok | df.d_C).values
    if nombre == "E A+C":
        return np.where(df.d_A, df.t_A, df.t_C), (df.d_A | df.d_C).values
    clave = nombre.split()[0]
    return df[f"t_{clave}"].values, df[f"d_{clave}"].values.astype(bool)


def main(trabajo: str) -> None:
    metodos = ["A bp+simb", "B audio", "C hibrido", "D consenso", "E A+C", "T techo"]
    filas, marcas, cats, comp = [], [], [], []
    todos = {m: [] for m in metodos}
    for f in sorted(glob.glob(os.path.join(trabajo, "por_nota_*.csv"))):
        pieza = os.path.basename(f)[len("por_nota_"):-4]
        df = pd.read_csv(f)
        for c in [c for c in df.columns if c.startswith("d_")]:
            df[c] = df[c].astype(bool)
        for m in metodos:
            t, d = combinar(df, m)
            err = (t - df.verdad.values) * 1000
            todos[m].append(err)
            filas.append({"pieza": pieza, "metodo": m, "pct_directas": 100 * d.mean(), **resumir(err)})
            if m in ("E A+C", "B audio", "A bp+simb"):
                for cat, msk in [("todas", np.ones(len(df), bool)), ("adornos (grace)", df.is_grace.values),
                                 ("con ornamento (trino, etc.)", df.adorno.values),
                                 ("en acorde", (df.acorde & ~df.is_grace).values), ("nota sola", (~df.acorde & ~df.is_grace).values),
                                 ("rubato fuerte (cuartil sup.)", df.rubato.values)]:
                    cats.append({"pieza": pieza, "metodo": m, "categoria": cat, "err": err[msk]})
                x = df.assign(t=t)
                x = x[x.onset_beat == x.groupby("compas_idx").onset_beat.transform("min")]  # primer ataque del compás
                g = x.groupby("compas_idx").agg(t=("t", "median"), v=("verdad", "median"))
                comp.append({"pieza": pieza, "metodo": m, "err": (g.t - g.v).values * 1000})
            if m == "E A+C":
                con_verdad = ~np.isnan(err)
                rev = (np.abs(df.t_A - df.t_B).values > 0.060)[con_verdad]
                grueso = np.abs(err[con_verdad]) > 100
                marcas.append({"pieza": pieza, "n": int(con_verdad.sum()), "pct_marcadas": 100 * rev.mean(),
                               "errores_gt100ms": int(grueso.sum()), "gruesos_marcados": int((rev & grueso).sum()),
                               "no_marcadas": int((~rev).sum()), "gruesos_no_marcados": int((grueso & ~rev).sum()),
                               "pct_errores_gt100ms_marcados": 100 * (rev & grueso).sum() / max(grueso.sum(), 1),
                               "pct_error_gt100ms_entre_no_marcadas": 100 * (grueso & ~rev).sum() / max((~rev).sum(), 1)})
    for m in metodos:
        filas.append({"pieza": "TOTAL", "metodo": m, **resumir(np.concatenate(todos[m]))})
    out = pd.DataFrame(filas)
    out.to_csv(os.path.join(AQUI, "resultados", "p05_combinaciones.csv"), index=False, float_format="%.2f")
    print(out[out.pieza == "TOTAL"].to_string())
    print(out.pivot(index="pieza", columns="metodo", values="pct_gt_200ms").round(1).to_string())
    c = pd.DataFrame(cats)
    tab = [{"metodo": m, "categoria": k, **resumir(np.concatenate(list(g.err)))} for (m, k), g in c.groupby(["metodo", "categoria"])]
    pd.DataFrame(tab).to_csv(os.path.join(AQUI, "resultados", "p05_por_categoria.csv"), index=False, float_format="%.2f")
    print(pd.DataFrame(tab).round(1).to_string())
    cp = pd.DataFrame(comp)
    tabc = [{"metodo": m, "pieza": "TOTAL", **resumir(np.concatenate(list(g.err)))} for m, g in cp.groupby("metodo")]
    tabc += [{"metodo": r.metodo, "pieza": r.pieza, **resumir(r.err)} for r in cp.itertuples()]
    pd.DataFrame(tabc).to_csv(os.path.join(AQUI, "resultados", "p05_por_compas.csv"), index=False, float_format="%.2f")
    print(pd.DataFrame(tabc).round(1).head(3).to_string())
    mk = pd.DataFrame(marcas)
    tot = {"pieza": "TOTAL", "n": int(mk.n.sum()),
           "pct_marcadas": float(np.average(mk.pct_marcadas, weights=mk.n)),
           "errores_gt100ms": int(mk.errores_gt100ms.sum()), "gruesos_marcados": int(mk.gruesos_marcados.sum()),
           "no_marcadas": int(mk.no_marcadas.sum()), "gruesos_no_marcados": int(mk.gruesos_no_marcados.sum())}
    tot["pct_errores_gt100ms_marcados"] = 100 * tot["gruesos_marcados"] / max(tot["errores_gt100ms"], 1)
    tot["pct_error_gt100ms_entre_no_marcadas"] = 100 * tot["gruesos_no_marcados"] / max(tot["no_marcadas"], 1)
    mk.loc[len(mk)] = tot
    mk.to_csv(os.path.join(AQUI, "resultados", "p05_marca_revisar.csv"), index=False, float_format="%.2f")
    print(mk.to_string())


if __name__ == "__main__":
    main(sys.argv[1])
