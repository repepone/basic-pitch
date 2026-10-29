"""Descarga los datos públicos usados en el laboratorio (no se suben al repo).

- (n)ASAP (CPJKU/asap-dataset, CC BY-NC-SA 4.0): partituras MusicXML, MIDI de
  interpretación y alineaciones nota a nota. Clonar a mano:
    git clone --filter=blob:none --no-checkout https://github.com/CPJKU/asap-dataset
  y hacer `git checkout HEAD -- <rutas>` de las obras de seleccion_asap.csv.
- MAESTRO v3.0.0 (CC BY-NC-SA 4.0): sólo los WAV/MIDI de las interpretaciones
  elegidas, leídos del zip remoto por rangos HTTP (sin bajar los 108 GB).

Uso: python bajar_datos.py <dir_asap> <dir_salida>
"""
import csv
import os
import sys

import pandas as pd
from remotezip import RemoteZip

MAESTRO_ZIP = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"
AQUI = os.path.dirname(os.path.abspath(__file__))


def main(dir_asap: str, dir_salida: str) -> None:
    meta = pd.read_csv(os.path.join(dir_asap, "metadata.csv"))
    sel = list(csv.DictReader(open(os.path.join(AQUI, "seleccion_asap.csv"))))
    os.makedirs(dir_salida, exist_ok=True)
    with RemoteZip(MAESTRO_ZIP) as z:
        for r in sel:
            fila = meta[meta.midi_performance == f"{r['folder']}/{r['performance']}.mid"].iloc[0]
            for col in ("maestro_audio_performance", "maestro_midi_performance"):
                rel = fila[col].replace("{maestro}/", "")
                ext = os.path.splitext(rel)[1]
                destino = os.path.join(dir_salida, f"{r['id']}{ext}")
                if os.path.exists(destino):
                    continue
                print("bajando", rel, flush=True)
                with z.open(f"maestro-v3.0.0/{rel}") as src, open(destino + ".part", "wb") as dst:
                    while True:
                        b = src.read(1 << 20)
                        if not b:
                            break
                        dst.write(b)
                os.rename(destino + ".part", destino)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
