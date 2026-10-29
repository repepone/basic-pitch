"""Receta: grabación humana + MusicXML → mapa «id de nota → ms» (y «compás → ms»).

    python receta_mapa.py grabacion.wav partitura.musicxml salida.json
                          [--repeticiones maximo|minimo] [--ventana-ms 120] [--desacuerdo-ms 60]

Pasos (ver INFORME-LAB-MAPA-HUMANO.md, apartado 4; es el método «E» medido en p05):
  1. partitura despliega las repeticiones (el músico decide cuáles se tocaron).
  2. basic-pitch (modelo ONNX del repo) transcribe la grabación a notas.
  3. Alineador A (simbólico): parangonar.DualDTWNoteMatcher entre la partitura y esas notas.
  4. Alineador B (audio): parangonar.AudioToScoreMatcher directamente sobre el audio; después
     «imán»: cada nota toma el onset de la nota de basic-pitch del mismo tono más cercana a B dentro
     de ±ventana (si no hay, se queda con B). Esto es C.
  5. Mapa final: A donde A casó la nota con una nota de basic-pitch; C en el resto.
  6. «revisar» = A y B discrepan más de --desacuerdo-ms (umbral elegido sobre las 8 piezas de p03).

Los ids son los atributos id de <note> del MusicXML (partitura los lee en importmusicxml.py:1335-1339);
con repeticiones desplegadas partitura añade el sufijo «-1», «-2»…, que aquí se separa en «repeticion».
"""
import argparse
import contextlib
import io
import json
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd
import soundfile as sf

import comun


def _silencio():
    return contextlib.redirect_stdout(io.StringIO())


def cargar(xml: str, repeticiones: str = "maximo", ids_interpretados: Optional[set] = None):
    import partitura as pt

    with _silencio():
        score = pt.load_musicxml(xml)
        part = pt.score.merge_parts(score.parts) if len(score.parts) > 1 else score.parts[0]
        ids = [n.id for n in part.notes_tied]
        if len(set(ids)) < len(ids) or None in ids:
            # MusicXML sin atributo id en <note> (p. ej. exportado por music21): ids provisionales por orden.
            # ¡No coincidirán con los del visor! Ver informe, apartado 4.
            print("AVISO: el MusicXML no trae ids únicos en <note>; se asignan n0, n1, ... por orden", file=sys.stderr)
            for j, n in enumerate(part.notes_tied):
                n.id = f"n{j}"
        unf = pt.score.unfold_part_maximal(part) if repeticiones == "maximo" else pt.score.unfold_part_minimal(part)
        na = unf.note_array(include_grace_notes=True)
    if ids_interpretados is not None:
        na = na[np.isin(na["id"], list(ids_interpretados))]
    medidas = sorted((m.start.t, m.number) for m in unf.iter_all(pt.score.Measure))
    inicios = np.array([m[0] for m in medidas])
    idx = np.searchsorted(inicios, na["onset_div"], side="right") - 1
    info = pd.DataFrame({
        "id": na["id"], "pitch": na["pitch"], "onset_beat": na["onset_beat"],
        "is_grace": na["is_grace"].astype(bool), "compas_idx": idx, "compas_num": [medidas[i][1] for i in idx],
    })
    return na, info, unf


def _interpolar(info: pd.DataFrame, tiempos: Dict[str, float]) -> np.ndarray:
    con = info[info.id.isin(tiempos.keys())].assign(t=lambda d: d.id.map(tiempos))
    g = con.groupby("onset_beat").t.median().sort_index()
    x, y = g.index.values, g.values
    keep = y >= np.maximum.accumulate(y) - 1e-9
    return np.interp(info.onset_beat.values, x[keep], y[keep])


def alineador_audio(wav: str, na, info) -> np.ndarray:
    import parangonar as pa

    y, sr = sf.read(wav)
    with _silencio():
        al = pa.AudioToScoreMatcher()(y, na, sample_rate=sr)
    d = {a["score_id"]: float(a["performance_time"]) for a in al if "performance_time" in a}
    t = np.array([d.get(i, np.nan) for i in info.id])
    return np.where(np.isnan(t), _interpolar(info, d), t)


def alineador_simbolico(na, info, notas, unf):
    import parangonar as pa
    import partitura as pt

    arr = np.array([(s, max(e - s, .01), p, int(max(1, min(127, round(v * 127)))), f"p{i}") for i, (s, e, p, v) in enumerate(notas)],
                   dtype=[("onset_sec", "f4"), ("duration_sec", "f4"), ("pitch", "i4"), ("velocity", "i4"), ("id", "U256")])
    with _silencio():
        perf = pt.performance.PerformedPart.from_note_array(arr).note_array()
        al = pa.DualDTWNoteMatcher()(na, perf, process_ornaments=True, score_part=unf)
    on = dict(zip(perf["id"], perf["onset_sec"].astype(float)))
    d = {a["score_id"]: on[a["performance_id"]] for a in al if a["label"] == "match"}
    t = _interpolar(info, d)
    return np.array([d.get(i, ti) for i, ti in zip(info.id, t)]), info.id.isin(d.keys()).values


def iman(guia: np.ndarray, info: pd.DataFrame, notas, ventana_ms: float):
    por_tono: Dict[int, list] = {}
    for s, _, p, _ in notas:
        por_tono.setdefault(p, []).append(s)
    por_tono = {p: np.array(sorted(v)) for p, v in por_tono.items()}
    t, directo = guia.copy(), np.zeros(len(info), bool)
    for k, (p, g) in enumerate(zip(info.pitch.values, guia)):
        c = por_tono.get(int(p))
        if c is None:
            continue
        j = np.searchsorted(c, g)
        cand = [c[i] for i in (j - 1, j) if 0 <= i < len(c)]
        m = min(cand, key=lambda x: abs(x - g))
        if abs(m - g) * 1000 <= ventana_ms:
            t[k], directo[k] = m, True
    return t, directo


def construir_mapa(wav: str, xml: str, repeticiones="maximo", ventana_ms=120.0, desacuerdo_ms=60.0,
                   ids_interpretados=None, cache=None) -> pd.DataFrame:
    na, info, unf = cargar(xml, repeticiones, ids_interpretados)
    notas = comun.notas(comun.salida_modelo(wav, cache))
    tA, dA = alineador_simbolico(na, info, notas, unf)
    tB = alineador_audio(wav, na, info)
    tC, dC = iman(tB, info, notas, ventana_ms)
    t = np.where(dA, tA, tC)
    revisar = np.abs(tA - tB) * 1000 > desacuerdo_ms
    return info.assign(ms=np.round(t * 1000).astype(int), directa=dA | dC, revisar=revisar)


def a_json(info: pd.DataFrame) -> dict:
    def partir(i):
        base, _, rep = i.rpartition("-")
        return (base, int(rep)) if rep.isdigit() and base else (i, 1)

    notas = []
    for r in info.sort_values("ms").itertuples():
        nid, rep = partir(r.id)
        notas.append({"id": nid, "repeticion": rep, "ms": int(r.ms), "evidencia": "directa" if r.directa else "interpolada",
                      "revisar": bool(r.revisar)})
    primero = info[info.onset_beat == info.groupby("compas_idx").onset_beat.transform("min")]  # primer ataque del compás
    comp = primero.groupby("compas_idx").agg(compas=("compas_num", "first"), ms=("ms", "median"), revisar=("revisar", "mean")).reset_index()
    compases = [{"compas": str(c.compas), "orden": int(c.compas_idx), "ms": int(c.ms), "revisar": bool(c.revisar > 0.5)}
                for c in comp.itertuples()]
    return {"notas": notas, "compases": compases}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("audio"); ap.add_argument("musicxml"); ap.add_argument("salida")
    ap.add_argument("--repeticiones", choices=["maximo", "minimo"], default="maximo")
    ap.add_argument("--ventana-ms", type=float, default=120.0)
    ap.add_argument("--desacuerdo-ms", type=float, default=60.0)
    a = ap.parse_args()
    info = construir_mapa(a.audio, a.musicxml, a.repeticiones, a.ventana_ms, a.desacuerdo_ms)
    json.dump(a_json(info), open(a.salida, "w"), indent=1, ensure_ascii=False)
    print(f"{len(info)} notas; {100 * info.directa.mean():.1f} % con evidencia directa; {100 * info.revisar.mean():.1f} % marcadas para revisar")
