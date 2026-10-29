"""Mapa «nota de la partitura → ms» sobre piano real (MAESTRO + partituras y alineaciones de (n)ASAP).

Métodos comparados (todos producen un tiempo para CADA nota de la partitura interpretada):
  A  bp+simb : notas de basic-pitch → parangonar.DualDTWNoteMatcher (simbólico) → onset de la nota casada;
               las no casadas se interpolan desde el mapa pulso→tiempo de las casadas.
  B  audio   : parangonar.AudioToScoreMatcher directo sobre el audio (sin basic-pitch); un tiempo por
               onset de partitura (resolución 20 ms).
  C  hibrido : B como guía + «imán»: cada nota toma el onset de la nota de basic-pitch del mismo tono más
               cercana dentro de ±VENTANA_MS; si no hay, se queda con B.
  T  techo   : DualDTWNoteMatcher con el MIDI real de la interpretación en vez de basic-pitch
               (transcripción perfecta): mide sólo el error del alineador.
Verdad: note_alignment.tsv de (n)ASAP (id de partitura → onset en el MIDI de la interpretación).

Uso: python p03_mapa_partitura.py <dir_asap> <dir_maestro_sel> <dir_trabajo> [ids...]
"""
import contextlib
import csv
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import soundfile as sf

import comun

AQUI = os.path.dirname(os.path.abspath(__file__))
VENTANA_MS = 120.0


def silencio():
    return contextlib.redirect_stdout(io.StringIO())


# ---------------------------------------------------------------- partitura
def cargar_partitura(xml: str, ids_interpretados: set):
    import partitura as pt

    with silencio():
        score = pt.load_musicxml(xml)
        part = pt.score.merge_parts(score.parts) if len(score.parts) > 1 else score.parts[0]
        unf = pt.score.unfold_part_maximal(part)
        na = unf.note_array(include_grace_notes=True)
    na = na[np.isin(na["id"], list(ids_interpretados))]
    # compás (índice secuencial en la partitura desplegada) y adornos por nota
    medidas = sorted(((m.start.t, m.end.t, m.number) for m in unf.iter_all(pt.score.Measure)))
    inicios = np.array([m[0] for m in medidas])
    idx_compas = np.searchsorted(inicios, na["onset_div"], side="right") - 1
    num_compas = np.array([medidas[i][2] for i in idx_compas])
    orn = {n.id: bool(n.ornaments) for n in unf.notes_tied}
    info = pd.DataFrame(
        {
            "id": na["id"],
            "pitch": na["pitch"],
            "onset_beat": na["onset_beat"],
            "is_grace": na["is_grace"].astype(bool),
            "adorno": [orn.get(i, False) for i in na["id"]],
            "compas_idx": idx_compas,
            "compas_num": num_compas,
        }
    )
    # acorde = varias notas no-adorno con el mismo onset en partitura
    tam = info[~info.is_grace].groupby("onset_beat").size()
    info["acorde"] = info.onset_beat.map(tam).fillna(1).astype(int) > 1
    return na, info, unf


def verdad(tsv: str) -> dict:
    d = {}
    for r in csv.DictReader(open(tsv), delimiter="\t"):
        if r["xml_id"].startswith("n") and r["midi_id"].startswith("n"):
            d[r["xml_id"]] = float(r["onset"])
    return d


def desfase(ref_maestro, ref_asap) -> float:
    """Desfase (s) que hay que sumar al MIDI de ASAP para caer en la base de tiempos del audio recortado.
    Normalmente 0; en alguna pieza ASAP recorta con otro origen (p. ej. Bach BWV 854: -0.5 s)."""
    cands = {round(a[0] - b[0], 4) for a in ref_maestro[:15] for b in ref_asap[:15] if a[2] == b[2]}

    def casadas(off):
        return comun.evaluar(ref_maestro, [(s + off, e + off, p, v) for s, e, p, v in ref_asap], 0.02)["casadas"]

    return max(sorted(cands | {0.0}), key=casadas)


def ids_tsv(tsv: str) -> set:
    return {r["xml_id"] for r in csv.DictReader(open(tsv), delimiter="\t") if r["xml_id"].startswith("n")}


# ---------------------------------------------------------------- utilidades de mapa
def perf_note_array(notas):
    import partitura as pt

    arr = np.array(
        [(s, max(e - s, 0.01), p, int(max(1, min(127, round(v * 127)))), f"p{i}") for i, (s, e, p, v) in enumerate(notas)],
        dtype=[("onset_sec", "f4"), ("duration_sec", "f4"), ("pitch", "i4"), ("velocity", "i4"), ("id", "U256")],
    )
    with silencio():
        pp = pt.performance.PerformedPart.from_note_array(arr)
    return pp.note_array()


def interpolar(info: pd.DataFrame, tiempos: dict) -> np.ndarray:
    """Para notas sin tiempo propio: interpola en pulsos desde la mediana por onset de las que sí tienen."""
    con = info[info.id.isin(tiempos.keys())].copy()
    con["t"] = con.id.map(tiempos)
    por_onset = con.groupby("onset_beat").t.median().sort_index()
    x, y = por_onset.index.values, por_onset.values
    # forzar monotonía: descartar puntos que retroceden respecto al máximo acumulado
    keep = y >= np.maximum.accumulate(y) - 1e-9
    x, y = x[keep], y[keep]
    if len(x) < 2:
        return np.full(len(info), np.nan)
    return np.interp(info.onset_beat.values, x, y)


def mapa_desde_alineacion(info, alineacion, perf_na) -> tuple:
    onset_perf = dict(zip(perf_na["id"], perf_na["onset_sec"].astype(float)))
    directos = {a["score_id"]: onset_perf[a["performance_id"]] for a in alineacion if a["label"] == "match"}
    t = interpolar(info, directos)
    t = np.array([directos.get(i, ti) for i, ti in zip(info.id, t)])
    return t, info.id.isin(directos.keys()).values


def metodo_simbolico(na_score, info, notas, unf):
    import parangonar as pa

    perf_na = perf_note_array(notas)
    with silencio():
        al = pa.DualDTWNoteMatcher()(na_score, perf_na, process_ornaments=True, score_part=unf)
    return mapa_desde_alineacion(info, al, perf_na)


def metodo_audio(wav, na_score, info):
    import parangonar as pa

    y, sr = sf.read(wav)
    with silencio():
        al = pa.AudioToScoreMatcher()(y, na_score, sample_rate=sr)
    d = {a["score_id"]: float(a["performance_time"]) for a in al if "performance_time" in a}
    t = np.array([d.get(i, np.nan) for i in info.id])
    if np.isnan(t).any():
        t = np.where(np.isnan(t), interpolar(info, {k: v for k, v in d.items()}), t)
    return t


def metodo_hibrido(guia, info, notas, ventana_ms=VENTANA_MS):
    por_tono = {}
    for s, e, p, v in notas:
        por_tono.setdefault(p, []).append(s)
    por_tono = {p: np.array(sorted(v)) for p, v in por_tono.items()}
    t = guia.copy()
    directo = np.zeros(len(info), bool)
    for k, (p, g) in enumerate(zip(info.pitch.values, guia)):
        c = por_tono.get(int(p))
        if c is None or np.isnan(g):
            continue
        j = np.searchsorted(c, g)
        cand = [c[i] for i in (j - 1, j) if 0 <= i < len(c)]
        mejor = min(cand, key=lambda x: abs(x - g))
        if abs(mejor - g) * 1000 <= ventana_ms:
            t[k] = mejor
            directo[k] = True
    return t, directo


# ---------------------------------------------------------------- métricas
def resumir(err_ms: np.ndarray) -> dict:
    e = err_ms[~np.isnan(err_ms)]
    if len(e) == 0:
        return {"n": 0}
    a = np.abs(e)
    return {
        "n": int(len(e)),
        "mediana_ms": float(np.median(e)),
        "p25_ms": float(np.percentile(e, 25)),
        "p75_ms": float(np.percentile(e, 75)),
        "mediana_abs_ms": float(np.median(a)),
        "p90_abs_ms": float(np.percentile(a, 90)),
        "pct_le_25ms": float(100 * np.mean(a <= 25)),
        "pct_le_50ms": float(100 * np.mean(a <= 50)),
        "pct_gt_200ms": float(100 * np.mean(a > 200)),
    }


def tempo_local(info, verdad_t):
    """|log| del cociente entre el tempo local (verdad) y el tempo mediano de la pieza, por onset."""
    df = info[~info.is_grace].copy()
    df["t"] = df.id.map(verdad_t)
    g = df.dropna().groupby("onset_beat").t.median().sort_index()
    b, t = g.index.values, g.values
    if len(b) < 3:
        return pd.Series(dtype=float)
    spb = np.diff(t) / np.maximum(np.diff(b), 1e-6)
    med = np.median(spb[spb > 0])
    r = np.abs(np.log(np.clip(spb, 1e-3, None) / med))
    return pd.Series(np.r_[r, r[-1]], index=b)


def main(asap, sel_dir, trabajo, ids=None):
    meta = pd.read_csv(os.path.join(asap, "metadata.csv"))
    sel = [r for r in csv.DictReader(open(os.path.join(AQUI, "seleccion_asap.csv"))) if not ids or r["id"] in ids]
    filas, filas_cat, filas_compas, chequeos = [], [], [], {}
    for r in sel:
        carpeta = os.path.join(asap, r["folder"])
        tsv = os.path.join(carpeta, f"{r['performance']}_note_alignments", "note_alignment.tsv")
        V = verdad(tsv)
        na, info, unf = cargar_partitura(os.path.join(carpeta, "xml_score.musicxml"), ids_tsv(tsv))
        wav = os.path.join(trabajo, f"{r['id']}_tramo.wav")
        out = comun.salida_modelo(wav, os.path.join(trabajo, f"{r['id']}_tramo_bp.npz"))
        bp = comun.notas(out)
        # control: el MIDI de ASAP y el de MAESTRO (recortado) comparten base de tiempos
        fila = meta[meta.midi_performance == f"{r['folder']}/{r['performance']}.mid"].iloc[0]
        ini = 0.0 if pd.isna(fila.start) else float(fila.start)
        fin = None if pd.isna(fila.end) else float(fila.end)
        ref_maestro = comun.notas_midi(os.path.join(sel_dir, f"{r['id']}.midi"), ini, fin)
        ref_asap = comun.notas_midi(os.path.join(carpeta, f"{r['performance']}.mid"))
        off = desfase(ref_maestro, ref_asap)
        ref_asap = [(s + off, e + off, p, v) for s, e, p, v in ref_asap]
        V = {k: v + off for k, v in V.items()}
        chequeos[r["id"]] = {"desfase_asap_s": off, **comun.evaluar(ref_maestro, ref_asap, 0.05)}

        verdad_v = np.array([V.get(i, np.nan) for i in info.id])
        resultados = {}
        resultados["A bp+simb"] = metodo_simbolico(na, info, bp, unf)
        tB = metodo_audio(wav, na, info)
        resultados["B audio"] = (tB, np.zeros(len(info), bool))
        resultados["C hibrido"] = metodo_hibrido(tB, info, bp)
        resultados["T techo (MIDI real)"] = metodo_simbolico(na, info, ref_asap, unf)

        rub = tempo_local(info, V)
        umbral_rub = rub.quantile(0.75) if len(rub) else np.inf
        info["rubato"] = info.onset_beat.map(rub).fillna(0) >= umbral_rub

        for nombre, (t, directo) in resultados.items():
            err = (t - verdad_v) * 1000
            filas.append({"pieza": r["id"], "metodo": nombre, "pct_directas": 100 * directo.mean(), **resumir(err)})
            for cat, m in [
                ("todas", np.ones(len(info), bool)),
                ("adornos (grace)", info.is_grace.values),
                ("con ornamento (trino, etc.)", info.adorno.values),
                ("en acorde", info.acorde.values & ~info.is_grace.values),
                ("nota sola", ~info.acorde.values & ~info.is_grace.values),
                ("rubato fuerte (cuartil sup.)", info.rubato.values),
                ("interpoladas (sin evidencia)", ~directo),
            ]:
                filas_cat.append({"pieza": r["id"], "metodo": nombre, "categoria": cat, **resumir(err[m])})
            # mapa por compás: primer onset del compás
            df = info.assign(t=t, v=verdad_v)
            g = df.groupby("compas_idx").agg(t=("t", "min"), v=("v", "min"))
            filas_compas.append({"pieza": r["id"], "metodo": nombre, **resumir((g.t - g.v).values * 1000)})
        # mapa de ejemplo (método C) para inspección
        por_nota = info.assign(verdad=verdad_v)
        for nombre, (t, directo) in resultados.items():
            clave = nombre.split()[0]
            por_nota[f"t_{clave}"] = t
            por_nota[f"d_{clave}"] = directo
        por_nota.to_csv(os.path.join(trabajo, f"por_nota_{r['id']}.csv"), index=False)
        tC, dC = resultados["C hibrido"]
        pd.DataFrame({"id": info.id.values, "compas": info.compas_num.values, "pitch": info.pitch.values,
                      "ms": pd.Series(tC * 1000).round().astype("Int64"), "evidencia_directa": dC,
                      "verdad_ms": pd.Series(verdad_v * 1000).round().astype("Int64")}).to_csv(
            os.path.join(trabajo, f"mapa_{r['id']}.csv"), index=False)
        print(pd.DataFrame([f for f in filas if f["pieza"] == r["id"]]).to_string(), flush=True)

    sufijo = "" if not ids else "_" + "_".join(ids)
    pd.DataFrame(filas).to_csv(os.path.join(AQUI, "resultados", f"p03_mapa_por_pieza{sufijo}.csv"), index=False, float_format="%.2f")
    pd.DataFrame(filas_cat).to_csv(os.path.join(AQUI, "resultados", f"p03_mapa_por_categoria{sufijo}.csv"), index=False, float_format="%.2f")
    pd.DataFrame(filas_compas).to_csv(os.path.join(AQUI, "resultados", f"p03_mapa_por_compas{sufijo}.csv"), index=False, float_format="%.2f")
    json.dump(chequeos, open(os.path.join(AQUI, "resultados", f"p03_chequeo_base_tiempos{sufijo}.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:] or None)
