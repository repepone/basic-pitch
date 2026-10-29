# Informe: laboratorio «mapa humano» con basic-pitch

Rama `lab-mapa-humano` del fork público `repepone/basic-pitch`, hecha para estudiar el código. No hay PR hacia el original. Fecha: 2026-10-29.

## Resumen en llano

1. basic-pitch (de Spotify, licencia Apache 2.0; los pesos del modelo vienen dentro del repo con la misma licencia) convierte audio en notas. Cuando encuentra una nota de piano, **su inicio es muy preciso**: la mitad de los errores son de unos 6 ms o menos y el límite teórico es de unos 12 ms.
2. El problema es que **no encuentra todas las notas**. Con su configuración de fábrica se le escapa un tercio de las notas de piano (sobre todo en acordes, pasajes rápidos y notas repetidas) y además inventa casi una de cada cuatro (muchas son armónicos). Cambiando un solo parámetro (`minimum_note_length` a 58 ms) las notas perdidas bajan a la mitad.
3. Al casar sus notas con la partitura (librerías libres `partitura` y `parangonar`) en 8 obras de piano reales, **el 88 % de las notas de la partitura queda a menos de 50 ms** de su momento real y un 4,5 % tiene errores grandes (más de 0,2 s). El alineador no es el problema: con el MIDI perfecto acierta el 99,5 %.
4. La receta marca automáticamente para revisar un 21 % de las notas, y ahí están el 94 % de los errores grandes. Es decir, **un humano tiene que repasar más o menos una nota de cada cinco**, además de decidir antes las repeticiones, recortar el audio a la obra y asegurar que los ids del MusicXML sean los del visor.
5. Fuera del piano sólo se pudo probar con instrumentos sintetizados (la red de este entorno bloqueaba los conjuntos de datos con grabaciones reales). En cuerda y conjunto el resultado es **mucho peor**: sólo la mitad de las notas a menos de 50 ms. Para esas familias, un mapa nota a nota automático no es fiable todavía.
6. La reverberación de la sala empeora el mapa: con una reverberación larga (2,5 s) los errores grandes se multiplican por entre 2,1 y 2,8 en las tres obras probadas.
7. Hay versión para navegador (basic-pitch-ts, con TF.js) y el modelo ONNX del repo también corre en el navegador; con ONNX en WebAssembly la inferencia va unas 17 veces más rápida que el tiempo real, mientras que TF.js sin GPU va más lento que el tiempo real. El alineado con la partitura sólo existe en Python.
8. Los datos usados para medir (MAESTRO, (n)ASAP) son de uso no comercial y no se han subido.

## Cómo leer este informe
- Cada afirmación lleva **leído** (sale de leer un fichero, con ruta y línea) o **medido** (sale de ejecutar un script de `lab_mapa_humano/`; la tabla de resultados está en `lab_mapa_humano/resultados/`).
- Error de onset = tiempo estimado − tiempo verdadero; positivo = llega tarde. «p90 |err|» = el 90 % de los errores es menor que esa cifra en valor absoluto.
- Entorno: contenedor Linux con 4 núcleos y sin GPU, Python 3.13, `onnxruntime` 1.29, `partitura` 1.9.0, `parangonar` 3.4.1, `librosa` 1.0.0, `mir_eval` 0.8.2. El modelo usado es `nmp.onnx` (TensorFlow no estaba instalado).
- Al final hay una lista **SIN MEDIR**.

## 0. Licencias

| Qué | Licencia | Dónde se ha leído | Marca |
|---|---|---|---|
| Código de basic-pitch (este fork) | Apache 2.0, «Copyright 2022 Spotify AB» | `LICENSE:1-3`, `README.md:213-220`, `pyproject.toml` (`license = { file = "LICENSE" }`), cabecera de cada `.py` | leído |
| Modelo (pesos `basic_pitch/saved_models/icassp_2022/`: `nmp/` TF, `nmp.onnx`, `nmp.tflite`, `nmp.mlpackage`) | No tiene licencia propia. Es el único `LICENSE` del repo y los pesos viajan dentro del paquete (`MANIFEST.in:4` los incluye), así que se distribuyen bajo la misma Apache 2.0 | `find -iname "*licen*"` sólo encuentra `./LICENSE`; `MANIFEST.in` | leído |
| Datos de entrenamiento | El repo no dice con qué datos se entrenó **el modelo publicado**. El código de entrenamiento menciona MAESTRO, GuitarSet, MedleyDB-Pitch, iKala y Slakh (`basic_pitch/constants.py:49-55`); varios de esos conjuntos tienen licencias no comerciales. El repo no aclara si eso afecta a los pesos | `constants.py:49-55` | leído |
| Rareza | `README.md:223` habla de «the Apache License or the GPL License», pero en el repo no hay ningún texto GPL | `README.md:223` | leído |
| `NOTICE` | Reconoce librosa (ISC), mir_eval (MIT), numpy (BSD), pretty-midi (MIT), resampy (ISC), scipy (BSD), tensorflow (Apache 2.0) y los audios de prueba de Vocadito (CC BY 4.0) | `NOTICE` | leído |
| basic-pitch-ts (versión navegador) | Apache 2.0; el modelo TF.js va en `model/` sin licencia aparte | `LICENSE` y `package.json:7` de spotify/basic-pitch-ts (commit 2d498f8); npm `@spotify/basic-pitch` 1.0.1 declara `Apache-2.0` | leído |
| partitura 1.9.0 / parangonar 3.4.1 | Apache 2.0 las dos | metadatos del paquete instalado (`License: Apache 2.0`; `License-Expression: Apache-2.0` + `licenses/LICENSE`) | leído |
| (n)ASAP (CPJKU/asap-dataset) | CC BY-NC-SA 4.0 | `LICENSE.md` del repo | leído |
| MAESTRO v3 | CC BY-NC-SA 4.0 | fichero `maestro-v3.0.0/LICENSE` dentro del zip oficial | leído |
| FluidR3_GM (síntesis) | MIT | `/usr/share/doc/fluid-soundfont-gm/copyright` | leído |
| Corales de Bach del corpus de music21 | dominio público (obras); music21 es BSD | no se verificó aquí | SIN MEDIR |

Consecuencia práctica: el código y el modelo se pueden usar y modificar (también en un producto) con Apache 2.0, conservando el aviso y el `NOTICE`. Los **datos** usados aquí para medir (MAESTRO, (n)ASAP) son **no comerciales**: sirven para estudiar, no para empaquetarlos. Ninguno se ha subido a la rama.

## 1. Cómo funciona por dentro

### 1.1 Entrada
- Cualquier formato que lea librosa; se carga **en mono y remuestreado a 22 050 Hz** (`basic_pitch/inference.py:239`, constante en `basic_pitch/constants.py:33`). — leído
- Se anteponen 3 840 muestras de silencio (`inference.py:242`) y el audio se trocea en ventanas de **43 844 muestras (≈1,99 s)** (`constants.py:47`) con salto de 36 164 muestras (≈1,64 s): 30 fotogramas de solape (`inference.py:302-305`, `194-219`). — leído
- Por ventana el modelo devuelve 172 fotogramas × 88 teclas de «onset» y de «nota» y 172 × 264 de «contorno» (tercios de semitono). Formas leídas del propio ONNX (`nmp.onnx`) y de `models.py:320`. — leído
- Al pegar las ventanas se tiran 15 fotogramas por lado (`inference.py:267-279`): quedan 142 útiles por ventana. — leído
- El modelo: CQT con salto de 256 muestras desde 27,5 Hz y 3 bins por semitono (`models.py:179-185`), «harmonic stacking» con armónicos 0,5, 1…7 (`models.py:213-218`) y tres cabezas convolucionales pequeñas (`models.py:241-318`). Rango de salida: 88 teclas, de La0 a Do8 (`constants.py:31-32`). Hay una capa de contorno comentada «porque se saltó sin querer al entrenar» (`models.py:226-238`). — leído

### 1.2 Resolución temporal real de los onsets
- Un fotograma son 256/22 050 s = **11,61 ms** (`constants.py:25,33`). El onset de una nota es **el índice del fotograma** donde la activación de onset tiene un máximo local (`note_creation.py:398-402`). No se interpola entre fotogramas. — leído
- El paso de fotograma a segundos (`note_creation.py:346-357`) aplica una corrección por ventana cada 172 fotogramas más un «número mágico» de 1,8 ms (`note_creation.py:47`); pero cada ventana aporta en realidad 142 fotogramas. Comparando con el tiempo exacto que sale del troceado, la librería se desvía en diente de sierra **entre −8,3 y +10,5 ms** (media +1,1 ms). — medido (`p00_resolucion_temporal.py`, `resultados/p00_resolucion.json`)
- Con 600 tonos armónicos sintéticos de ataque exacto (posición sub-fotograma aleatoria): error de onset **mediana −2,3 ms, cuartiles −5,8 / +1,1 ms, 90 % de los errores ≤ 9,8 ms en valor absoluto, extremos −15,8 / +12,6 ms**. La mediana se va de −0,4 ms al principio de cada ventana a −4,6 ms al final: es el diente de sierra. Usando el mapeo exacto en lugar del de la librería, la dispersión baja algo (cuartiles −6,2 / −0,3 ms; p90 8,5 ms). — medido (mismo script)
- Conclusión: **el suelo de precisión es de ±1 fotograma (~12 ms)**; cualquier mapa construido con basic-pitch no puede ser más fino que eso. — deducido de lo leído y medido arriba

### 1.3 Umbrales
| Parámetro (CLI) | Por defecto | Qué hace | Dónde |
|---|---|---|---|
| `onset_threshold` (`--onset-threshold`) | 0,5 | Un máximo local de la activación de onset sólo abre nota si vale ≥ umbral. Antes, la activación de onset se sustituye por el máximo entre ella y la subida de la activación de nota en 1-2 fotogramas («onsets inferidos») | `inference.py:185`; `note_creation.py:395-402`, `289-311` |
| `frame_threshold` (`--frame-threshold`) | 0,3 | La nota sigue viva mientras la activación de nota ≥ umbral; se cierra tras 11 fotogramas seguidos por debajo (`energy_tol`) | `inference.py:186`; `note_creation.py:416-426` |
| `minimum_note_length` (`--minimum-note-length`) | 127,7 ms | Se pasa a fotogramas redondeando (127,7 ms → 11) y se **descartan las notas de ≤ 11 fotogramas**, es decir, la duración mínima efectiva es 12 fotogramas ≈ **139 ms** | `inference.py:187`, `469`; `note_creation.py:429`, `496` |
| `melodia_trick` (`--no-melodia` lo apaga) | activado | Después de las notas con onset, busca energía de nota sobrante y la convierte en notas **sin onset detectado** | `note_creation.py:449-509` |
| `minimum/maximum_frequency` | ninguno | Pone a cero las teclas fuera de rango | `note_creation.py:314-343` |

Detalle importante leído en `note_creation.py:411-436`: las notas se procesan de la última a la primera y cada nota «gasta» la energía de su tecla y de las dos vecinas (±1 semitono). Por eso una nota repetida del mismo tono queda cortada donde empieza la siguiente, y si la distancia entre ambas es menor que la duración mínima, se tira. — leído (efecto medido en el apartado 2.4)

### 1.4 Polifonía
- 88 salidas independientes por fotograma: **no hay límite de notas simultáneas** (`models.py:282-318`; el decodificado recorre todos los picos de onset, `note_creation.py:411-447`). — leído
- Limitaciones de diseño: borrar la energía de los semitonos vecinos (`note_creation.py:432-436`, `466-469`) debería penalizar los clústeres de segundas menores (deducción, SIN MEDIR); el README dice que «funciona mejor con un instrumento a la vez» (`README.md:14`). — leído

### 1.5 ¿Hay versión para navegador y qué cuesta?
- **basic-pitch-ts** (spotify/basic-pitch-ts, Apache 2.0): TypeScript + **TF.js** (`package.json:53`, `@tensorflow/tfjs ^3.2.0`). Modelo TF.js: `model.json` 174 537 B + `group1-shard1of1.bin` 742 392 B. Exige un `AudioBuffer` **ya remuestreado a 22 050 Hz y mono** (`src/inference.ts:169-182`): el remuestreo lo pone quien lo usa. Su decodificado por defecto usa `minNoteLen = 5` fotogramas (`src/toMidi.ts:347`), no 11 como Python, y su README de ejemplo usa umbrales 0,25 / 0,25. En npm: `@spotify/basic-pitch` 1.0.1. — leído
- **ONNX**: este repo trae `nmp.onnx` (230 444 B), que corre con `onnxruntime-web`. No hay un paquete oficial de basic-pitch para ONNX en web: habría que portar el troceado (`inference.py`) y el decodificado (`note_creation.py`) o reutilizar los de basic-pitch-ts. — leído
- Tamaños de descarga (sin comprimir / gzip), medidos con `ls` y `gzip`: `tf.min.js` 4.22.0 → 1 469 843 B / 368 002 B; `ort.wasm.min.js` 1.20.1 → 49 026 B / 15 457 B + `ort-wasm-simd-threaded.wasm` → 11 246 032 B / 2 977 104 B; `nmp.onnx` → 111 625 B en gzip; pesos TF.js → 427 393 B en gzip. — medido
- Velocidad: ver apartado 2.5.

## 2. Precisión de los onsets frente a una verdad conocida

Todas las cifras: notas casadas por tono exacto y onset dentro de ±50 ms (`mir_eval`), error = onset de basic-pitch − onset verdadero (positivo = basic-pitch llega tarde). «Perdidas» = notas reales sin pareja; «inventadas» = notas de basic-pitch sin pareja. Umbrales por defecto salvo que se diga otra cosa. Modelo: `nmp.onnx` del repo (TensorFlow no está instalado aquí; el ONNX es una conversión del modelo TF, `README.md:53`).

### 2.1 Piano real (MAESTRO v3, verdad = MIDI del Disklavier) — medido
8 interpretaciones, tramos de las obras que usa (n)ASAP, 1 210 s de audio en total (`p01_onsets_piano_maestro.py` → `resultados/p01_piano_por_pieza.csv`).

| pieza | rasgo | notas reales | notas bp | % perdidas | % inventadas | mediana (ms) | cuartiles (ms) | p90 \|err\| (ms) | % \|err\| ≤ 20 ms |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Bach, Preludio BWV 860 | semicorcheas rápidas | 611 | 403 | 41,7 | 11,7 | +4,4 | −2,1 / +9,7 | 21,1 | 88,2 |
| Bach, Fuga BWV 854 | contrapunto | 734 | 655 | 25,8 | 16,8 | +1,9 | −3,4 / +7,5 | 15,9 | 93,6 |
| Chopin, Estudio op. 10 n.º 2 | cromatismo rápido | 1 391 | 468 | 69,7 | 10,0 | +2,9 | −3,3 / +8,5 | 22,6 | 87,6 |
| Schubert, Moment musical n.º 3 | melodía + acordes | 1 034 | 931 | 26,6 | 18,5 | −1,0 | −6,4 / +5,0 | 16,0 | 93,7 |
| Haydn, Sonata para teclado 31, 1.er mov. (numeración de ASAP) | adornos | 1 609 | 1 337 | 31,2 | 17,2 | 0,0 | −5,2 / +4,9 | 13,8 | 95,3 |
| Mozart, Sonata para piano n.º 12, 1.er mov. (numeración de ASAP) | adornos + escalas | 2 506 | 2 243 | 21,4 | 12,2 | +5,2 | −0,6 / +10,3 | 16,9 | 93,7 |
| Rajmáninov, Preludio op. 23 n.º 6 | textura densa con pedal | 1 248 | 1 365 | 23,9 | 30,4 | −2,0 | −8,4 / +3,3 | 17,7 | 91,9 |
| Debussy, Reflets dans l'eau | pedal, resonancia | 2 019 | 2 126 | 41,4 | 44,3 | −1,3 | −7,3 / +4,3 | 17,0 | 93,1 |
| **Total** | | **11 152** | **9 528** | **34,6** | **23,5** | **+1,2** | **−4,8 / +7,2** | **17,0** | **93,0** |

Lectura: cuando basic-pitch encuentra la nota, **el onset es muy bueno** (la mitad de los errores dentro de ±6 ms). El problema es **cuántas notas no encuentra**: un tercio con los valores por defecto, y dos tercios en un pasaje rápido como el Estudio de Chopin. Ampliar la tolerancia a 100 ms apenas cambia (33,0 % perdidas, 21,5 % inventadas; `resultados/p01_piano_tolerancias.json`): las que faltan no están desplazadas, **no están**. — medido

Por qué faltan y qué sobra (`p08_tipos_de_error.py` → `resultados/p08_tipos_de_error.csv`), total piano real: **62 % de las perdidas están dentro de un acorde** (otra nota real empieza a ±30 ms); de las inventadas, **52 % son «armónicos»** (hay una nota real sonando a −12, +12, +19, +24, +28 o +31 semitonos), 18 % son re-ataques falsos de una nota que sigue sonando y 30 % otras. — medido

### 2.2 Umbrales (barrido sobre las 8 piezas de piano real) — medido
`resultados/p01_piano_barrido_umbrales.csv` (30 combinaciones). Extracto:

| onset_threshold | frame_threshold | minimum_note_length | % perdidas | % inventadas | mediana (ms) | p90 \|err\| (ms) |
|--:|--:|--:|--:|--:|--:|--:|
| 0,5 | 0,3 | 127,7 ms (defecto) | 34,6 | 23,5 | +1,2 | 17,0 |
| 0,5 | 0,3 | 58 ms | 18,6 | 27,1 | +2,0 | 15,4 |
| 0,6 | 0,3 | 58 ms | 19,9 | 21,3 | +2,0 | 15,2 |
| 0,5 | 0,2 | 58 ms | 12,4 | 38,3 | +2,0 | 15,8 |
| 0,7 | 0,4 | 127,7 ms | 46,7 | 11,6 | +1,1 | 16,7 |
| 0,3 | 0,2 | 58 ms | 14,6 | 62,9 | +2,3 | 17,8 |

- **`minimum_note_length` es el mando que más importa**: bajarlo de 127,7 a 58 ms casi divide por dos las perdidas (34,6 → 18,6 %) a cambio de pocas inventadas más (23,5 → 27,1 %). Con 0,6 / 0,3 / 58 ms se pierden 19,9 % y se inventan 21,3 %: mejor en las dos cosas que el valor por defecto. — medido
- `onset_threshold` y `frame_threshold` intercambian perdidas por inventadas; la precisión temporal casi no cambia en todo el barrido (mediana entre +0,6 y +2,3 ms). — medido
- `melodia_trick` apagado (`--no-melodia`): perdidas 34,6 → 37,3 %, inventadas 23,5 → 18,4 % (`resultados/p04_repetidas_y_melodia.json`). — medido

### 2.3 Por familia de instrumento (audio sintetizado desde MIDI) — medido, con reservas
No se pudo bajar audio real de otras familias: Zenodo, Hugging Face, archive.org y el servidor de Vienna 4x22 están **bloqueados por la red de este entorno** (403 del proxy), así que no se pudieron usar URMP, Bach10, GuitarSet, MusicNet ni similares (medido: 403 del proxy). En su lugar: 10 corales de Bach (corpus de music21) con tempo «humanizado» (rubato ±15 % y ±10 ms de jitter), sintetizados con FluidR3_GM, en seco y con reverberación fuerte de fluidsynth (`p02_onsets_sintetizado.py`, `p02b_tolerancia.py`).

**Aviso importante**: la «verdad» es el note-on del MIDI. En viento y cuerda la muestra tarda en sonar, así que un retraso de 15-35 ms es en parte del sintetizador y no de basic-pitch. Y las muestras de FluidR3 no son instrumentos humanos: vibrato, arco y ataques son más simples. Estas cifras sirven para comparar familias entre sí, no como precisión real en violín o flauta. (El retraso propio de cada muestra no se midió aparte: SIN MEDIR.)

Seco, tolerancia ±50 ms (`resultados/p02_sintetizado_por_familia.csv`) y, entre paréntesis, perdidas con ±100 ms (`resultados/p02b_tolerancia.csv`):

| familia | notas reales | % perdidas (±100 ms) | % inventadas | mediana (ms) | cuartiles (ms) | p90 \|err\| (ms) |
|:--|--:|--:|--:|--:|--:|--:|
| piano (FluidR3), 4 voces | 2 423 | 11,9 (11,5) | 25,5 | +6,8 | +1,9 / +12,2 | 17,1 |
| solo: flauta | 543 | 23,6 (1,1) | 29,9 | +30,1 | +24,8 / +35,1 | 39,8 |
| solo: clarinete | 543 | 5,9 (0,9) | 69,2 | +21,4 | +13,8 / +27,9 | 36,0 |
| solo: violín | 543 | 32,2 (24,5) | 52,7 | +15,1 | +8,2 / +21,4 | 26,0 |
| solo: violonchelo | 651 | 29,6 (5,5) | 44,8 | +21,0 | +12,7 / +31,8 | 42,8 |
| solo: trompeta | 543 | 0,0 (0,0) | 0,0 | +13,2 | +9,1 / +17,1 | 21,8 |
| cuerda: cuarteto | 2 423 | 44,9 (25,3) | 67,9 | +27,5 | +19,4 / +36,5 | 43,6 |
| cuerda: «String Ensemble» | 2 423 | 67,7 (36,9) | 84,1 | +27,6 | +16,9 / +37,7 | 44,6 |
| conjunto: maderas | 2 423 | 26,0 (14,6) | 48,0 | +19,7 | +7,4 / +30,1 | 39,4 |
| conjunto: vl + cl + trompa + vc | 2 423 | 45,1 (19,3) | 62,4 | +27,2 | +19,5 / +36,0 | 43,8 |

- Fuera del piano los onsets salen **tarde** (mediana entre +13 y +30 ms, frente a +1,2 ms en piano real y +6,8 ms en piano sintetizado) y **sobran muchas notas**. En la cuerda y los solos de cuerda casi todas las inventadas (55-99 %) son **la misma nota sostenida partida en varias** (re-ataques falsos); en el clarinete, el 75 % de las inventadas son armónicos (`resultados/p08_tipos_de_error.csv`). — medido
- La reverberación de fluidsynth empeora sobre todo perdidas e inventadas (por ejemplo, el violonchelo pasa de 29,6 a 47,9 % de perdidas y el cuarteto de 44,9 a 58,8 %), no la mediana del onset. — medido

### 2.4 Notas repetidas — medido
Trenes de 24 notas iguales a distintas distancias (`p04_repetidas_y_melodia.py`, piano y violín FluidR3, 72 notas por caso):

| distancia entre notas | piano, defecto (127,7 ms) | piano, 58 ms | violín, defecto | violín, 58 ms |
|--:|--:|--:|--:|--:|
| 60 ms | 68,1 % perdidas | 47,2 % | 86,1 % | 81,9 % |
| 80 ms | 51,4 % | 5,6 % | 77,8 % | 62,5 % |
| 100 ms | 50,0 % | 0,0 % | 68,1 % | 59,7 % |
| 120 ms | 45,8 % | 0,0 % | 80,6 % | 70,8 % |
| 140 ms | 11,1 % | 0,0 % | 76,4 % | 76,4 % |
| 160 ms | 0,0 % | 0,0 % | 83,3 % | 83,3 % |
| 200 ms | 0,0 % | 0,0 % | 63,9 % | 58,3 % |

Confirma lo leído en el código (apartado 1.3): **en piano, con el valor por defecto, se pierde una de cada dos notas repetidas a menos de ~140 ms**; con 58 ms desaparece el problema a partir de 100 ms. En el violín sintetizado se pierde más de la mitad de las repeticiones a todas las velocidades probadas, también con 58 ms. — medido

### 2.5 Coste en el navegador — medido
`lab_mapa_humano/navegador/bench.html` + `bench.js` en el Chromium headless de este entorno (Playwright), máquina sin otros procesos, sólo **inferencia del modelo** con audio aleatorio, 30 ventanas tras 3 de calentamiento (`resultados/navegador_bench.json`). Cada ventana aporta 1,64 s de audio nuevo:

| motor | carga del modelo | mediana por ventana | velocidad |
|:--|--:|--:|--:|
| ONNX (`onnxruntime-web` 1.20.1, WASM, 1 hilo) | 781 ms | 93,2 ms | **17,6× tiempo real** |
| TF.js 4.22.0, backend `cpu` (JavaScript puro) con el modelo de basic-pitch-ts | 29 ms | 3 810 ms | 0,4× (más lento que tiempo real) |
| TF.js 4.22.0, backend `webgl` | 27 ms | 3 396 ms | 0,5× |

- Con ONNX + WASM, una obra de 5 minutos tardaría unos 17 s sólo en la inferencia (extrapolado de la tabla). — medido / extrapolado
- El «WebGL» de este Chromium headless va por software (SwiftShader), así que la fila de WebGL **no** representa un navegador con GPU. Tampoco se probó el backend `wasm` de TF.js. — SIN MEDIR
- Como referencia, en Python (`onnxruntime` 1.29, 4 núcleos) los 273,2 s de Mozart tardaron 5,8 s de inferencia más 4,1 s de decodificado a notas. — medido
- Lo que **no** existe en el navegador: parangonar y partitura son Python; el alineado con la partitura habría que hacerlo en un servidor o portarlo. — leído

## 3. Casar la salida con una partitura

### 3.1 Montaje — leído / medido
- **Obras y verdad**: las 8 interpretaciones del apartado 2.1. Partituras MusicXML y alineación nota a nota partitura ↔ interpretación de **(n)ASAP** (`note_alignment.tsv`, hecha por sus autores con parangonar y revisada a mano según su README). La verdad de cada nota de la partitura es el onset de su nota en el MIDI del Disklavier. Se comprueba que el MIDI de ASAP y el de MAESTRO recortado coinciden nota a nota; en la Fuga BWV 854 el de ASAP está **desplazado +0,5 s** (734/734 notas casan sólo con −0,5 s) y se corrige (`desfase()` en `p03_mapa_partitura.py`, `resultados/p03_chequeo_base_tiempos.json`). — medido
- **Repeticiones**: se despliega la partitura con `unfold_part_maximal` de partitura y se queda sólo con las notas que (n)ASAP da por interpretadas. Es decir, **la estructura de repeticiones se da por conocida**. — leído
- **Métodos** (todos dan un tiempo para cada nota de la partitura; `p03_mapa_partitura.py`, `p05_combinaciones.py`):
  - **A** basic-pitch → `parangonar.DualDTWNoteMatcher` (alineador simbólico nota a nota). Notas sin pareja: interpolación en pulsos desde las casadas.
  - **B** `parangonar.AudioToScoreMatcher`: alinea la partitura **directamente con el audio** (espectrograma y onsets propios, 50 fotogramas/s, pensado para piano). No usa basic-pitch.
  - **C** B + «imán»: cada nota toma el onset de la nota de basic-pitch del mismo tono más cercana dentro de ±120 ms.
  - **E** (la receta): A donde A casó la nota con una nota de basic-pitch; C en el resto.
  - **T (techo)**: el mismo alineador simbólico que A, pero con el MIDI real en lugar de basic-pitch. Mide sólo lo que se equivoca el alineador.

### 3.2 Resultados (10 756 notas de partitura con verdad) — medido
`resultados/p05_combinaciones.csv`:

| método | mediana (ms) | cuartiles (ms) | mediana \|err\| (ms) | p90 \|err\| (ms) | % ≤ 25 ms | % ≤ 50 ms | % > 200 ms |
|:--|--:|--:|--:|--:|--:|--:|--:|
| A basic-pitch + parangonar simbólico | +1,5 | −7,3 / +10,2 | 8,9 | 90,2 | 78,2 | 86,5 | 5,6 |
| B parangonar audio directo | −6,2 | −20,2 / +10,0 | 16,2 | 76,1 | 67,0 | 85,2 | 5,5 |
| C = B + imán basic-pitch | −0,1 | −10,3 / +8,4 | 9,2 | 71,6 | 78,6 | 87,6 | 5,5 |
| **E = A, y si no C (receta)** | **+0,2** | **−9,2 / +8,5** | **8,8** | **65,6** | **79,8** | **88,2** | **4,5** |
| T techo (MIDI real + parangonar) | 0,0 | 0,0 / 0,0 | 0,0 | 0,0 | 99,5 | 99,5 | 0,2 |

Por pieza, método E:

| pieza | n | mediana \|err\| (ms) | p90 \|err\| (ms) | % ≤ 50 ms | % > 200 ms |
|:--|--:|--:|--:|--:|--:|
| Bach, Fuga BWV 854 | 731 | 6,7 | 28,4 | 94,1 | 1,1 |
| Bach, Preludio BWV 860 | 602 | 9,2 | 36,6 | 92,2 | 2,3 |
| Chopin, Estudio op. 10 n.º 2 | 1 376 | 13,0 | 77,8 | 86,7 | 2,6 |
| Debussy, Reflets dans l'eau | 1 859 | 11,7 | 206,5 | 77,2 | 10,4 |
| Haydn, Sonata 31, 1.er mov. | 1 498 | 7,0 | 58,4 | 89,4 | 4,8 |
| Mozart, Sonata n.º 12, 1.er mov. | 2 448 | 7,9 | 25,7 | 94,9 | 2,2 |
| Rajmáninov, op. 23 n.º 6 | 1 220 | 8,8 | 88,4 | 85,2 | 5,3 |
| Schubert, Moment musical n.º 3 | 1 022 | 7,8 | 50,8 | 89,4 | 4,4 |

- **El alineador no es el cuello de botella**: con el MIDI real (T) parangonar acierta el 99,5 % de las notas a ≤ 25 ms. Todo el error de A, C y E viene de que basic-pitch **no ve una de cada tres notas** y de las notas que inventa. — medido
- Ningún método gana en todas las obras: B (audio directo) falla mucho en Rajmáninov (13,8 % > 200 ms) y A (simbólico) en Haydn (6,6 %). Combinarlos (E) da el mejor total, pero **la elección de E se hizo mirando estas mismas 8 piezas**: es exploratoria, no validada en obras nuevas. — medido
- Sólo el 72,5 % de las notas de la partitura recibe un onset de basic-pitch («evidencia directa»); el resto se interpola. — medido

**Mapa por compás** (tiempo = mediana de las notas del primer ataque del compás; 600 compases; `resultados/p05_por_compas.csv`): con E, mediana |err| 8,4 ms, p90 51,7 ms, 89,7 % ≤ 50 ms, 3,8 % > 200 ms. El simbólico A da aquí un poco mejor (90,2 % ≤ 50 ms; 3,3 % > 200 ms). — medido

### 3.3 Dónde falla — medido salvo donde se indica
Método E, por tipo de nota (`resultados/p05_por_categoria.csv`):

| tipo de nota | n | mediana \|err\| (ms) | p90 \|err\| (ms) | % ≤ 50 ms | % > 200 ms |
|:--|--:|--:|--:|--:|--:|
| nota sola | 3 649 | 7,8 | 50,5 | 89,9 | 3,7 |
| en acorde | 7 063 | 9,3 | 72,1 | 87,5 | 5,0 |
| rubato fuerte (25 % de onsets con más cambio de tempo local) | 2 613 | 9,2 | 101,8 | 85,0 | 6,8 |
| apoyaturas / notas de adorno (grace) | 44 | 14,9 | 137,6 | 70,4 | 4,6 |
| nota con ornamento (trino, mordente…) | 35 | 9,6 | 187,7 | 71,4 | 8,6 |

- **Adornos**: peor que el resto (70-71 % ≤ 50 ms), pero con **muy pocos casos (n = 44 y 35)**: la cifra es orientativa. Un trino de la partitura es una sola nota y en la grabación son muchas; qué nota del trino elige el alineador no se ha comprobado (SIN MEDIR). — medido (n pequeño)
- **Acordes**: algo peor que las notas solas (5,0 % frente a 3,7 % > 200 ms). Además, el 62 % de las notas que basic-pitch pierde en piano están en acordes (apartado 2.1). En la interpretación humana las notas de un acorde no suenan a la vez (arpegiados, melodía adelantada); el mapa da a cada nota la suya cuando la encuentra. — medido
- **Rubato**: donde el tempo cambia más, los errores gruesos casi se duplican (6,8 % > 200 ms frente al 4,5 % general). — medido
- **Pedal y resonancia**: Debussy es la peor obra (10,4 % > 200 ms; 39 % de notas marcadas para revisar) y Rajmáninov la segunda. En las dos basic-pitch inventa más notas (44 % y 30 %). — medido
- **Reverberación añadida**: medido con respuesta al impulso sintética (ruido con caída exponencial, 50 % seco / 50 % húmedo) sobre Schubert, Haydn y Mozart (`p06_reverberacion.py` → `resultados/p06_reverberacion.csv`):

| pieza | RT60 | onsets bp: % perdidas / % inventadas | E: % ≤ 50 ms | E: % > 200 ms | B (audio directo): % ≤ 50 ms / % > 200 ms | E: % marcadas | E: % > 100 ms entre no marcadas |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Schubert | 0 (original) | 26,6 / 18,5 | 89,4 | 4,4 | 88,9 / 3,2 | 15,7 | 0,6 |
| Schubert | 1,0 s | 35,2 / 28,1 | 80,8 | 7,0 | 83,2 / 4,9 | 25,3 | 0,9 |
| Schubert | 2,5 s | 38,2 / 38,4 | 73,5 | 12,5 | 79,4 / 6,6 | 34,6 | 2,4 |
| Haydn | 0 (original) | 31,2 / 17,2 | 89,4 | 4,8 | 93,8 / 1,9 | 17,0 | 0,5 |
| Haydn | 1,0 s | 42,0 / 30,8 | 79,5 | 5,9 | 85,4 / 2,0 | 30,1 | 1,5 |
| Haydn | 2,5 s | 44,7 / 38,4 | 75,1 | 11,7 | 82,0 / 4,9 | 38,1 | 2,6 |
| Mozart | 0 (original) | 21,4 / 12,2 | 94,9 | 2,2 | 92,8 / 1,1 | 10,1 | 0,1 |
| Mozart | 1,0 s | 29,5 / 19,3 | 88,6 | 2,4 | 90,4 / 0,8 | 16,1 | 1,0 |
| Mozart | 2,5 s | 33,0 / 27,4 | 85,1 | 4,7 | 87,6 / 0,9 | 21,2 | 0,9 |

  La reverberación **no mueve la mediana del onset** (sigue entre −1 y +5 ms), pero basic-pitch pierde e inventa más notas, y el mapa E empeora: con 2,5 s de RT60 los errores > 200 ms se multiplican por 2,8 (Schubert), 2,4 (Haydn) y 2,1 (Mozart). **El alineador de audio directo (B) aguanta mejor la reverberación que E** (en las 9 condiciones tiene menos errores > 200 ms). La marca «revisar» sigue funcionando razonablemente (≤ 2,6 % de errores > 100 ms sin marcar), pero marca más notas (hasta el 38 %). — medido
- **Fragmentos rápidos**: en el Estudio de Chopin basic-pitch pierde el 70 % de las notas y el mapa se apoya en interpolación; aun así queda 86,7 % ≤ 50 ms porque el ritmo de semicorcheas es regular. — medido
- **Principio de la obra**: el alineador de audio (B) pone el primer ataque de la partitura en 0,0 s en las 8 piezas (en 7 de ellas el ataque real está entre 0,6 y 1,1 s). E lo corrige cuando basic-pitch ve esas notas; en el Estudio de Chopin 15 de los 36 errores > 200 ms de E están en el primer compás. — medido
- **Errores agrupados**: los errores > 200 ms se concentran en pocos compases (por ejemplo, en Debussy 20 de 193 en un mismo compás). Revisar por compases es más eficaz que nota a nota. — medido

### 3.4 Fuera del piano (cuerda y conjunto sintetizados) — medido, con las reservas de 2.3
Los mismos 10 corales de 2.3 convertidos a MusicXML y alineados con su audio sintetizado (`p07_mapa_no_piano.py` → `resultados/p07_mapa_no_piano.csv`; 2 408 notas de partitura por caso). Método E de la receta:

| conjunto | reverb | mediana (ms) | cuartiles (ms) | mediana \|err\| (ms) | % ≤ 50 ms | % > 200 ms | % marcadas | % > 100 ms entre no marcadas |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| piano (FluidR3) | no | +6,2 | +1,0 / +12,3 | 7,7 | 96,2 | 2,8 | 9,0 | 0,0 |
| piano (FluidR3) | sí | +5,6 | +0,5 / +11,7 | 7,1 | 95,4 | 3,5 | 13,5 | 0,0 |
| cuarteto de cuerda | no | +39,7 | +23,4 / +83,8 | 43,8 | 55,4 | 10,0 | 40,3 | 5,6 |
| cuarteto de cuerda | sí | +66,7 | +29,3 / +157,6 | 80,3 | 36,8 | 22,7 | 60,1 | 19,6 |
| vl + cl + trompa + vc | no | +40,9 | +24,8 / +79,4 | 42,4 | 56,7 | 3,0 | 31,8 | 5,0 |
| vl + cl + trompa + vc | sí | +47,8 | +26,3 / +101,1 | 51,5 | 48,8 | 13,5 | 48,3 | 7,1 |

- En **piano sintetizado** la receta funciona igual o mejor que en piano real (96 % ≤ 50 ms) y la marca «revisar» deja 0,0 % de errores > 100 ms sin marcar. — medido
- En **cuerda y conjunto** el mapa sale **sistemáticamente tarde** (mediana +40 ms) y sólo la mitad de las notas cae a ≤ 50 ms. Parte de ese retraso es el ataque lento de las muestras de FluidR3 frente al note-on del MIDI (ver 2.3), así que la cifra no se puede trasladar tal cual a un cuarteto real. — medido, con reservas
- Con reverberación, en el cuarteto **la marca «revisar» deja de servir**: marca el 60 % de las notas y aun así quedan errores > 100 ms en el 19,6 % de las no marcadas. — medido
- Ningún método es claramente mejor fuera del piano (A, B, C y E dan entre 46 y 62 % ≤ 50 ms en seco). El alineador de audio de parangonar está pensado para piano (`AudioToScoreMatcher`: 88 bandas de A0 a C8, «one per piano key», en su docstring). — medido / leído

**Conclusión para cuerda y conjunto**: con esta receta, un mapa **nota a nota** necesitaría revisión humana de entre un tercio y más de la mitad de las notas. No se midió aquí si el mapa por compás aguanta mejor fuera del piano, ni nada con grabaciones reales (SIN MEDIR).

## 4. Receta: grabación humana + MusicXML → mapa `id de nota → ms`

Script: `lab_mapa_humano/receta_mapa.py` (método E del apartado 3, con la marca «revisar»).

### 4.1 Pasos
1. **Instalar** (Python ≥ 3.10): este repo (`pip install -e .`), `onnxruntime`, `partitura`, `parangonar`, `soundfile`, `tqdm` (parangonar 3.4.1 lo importa pero no lo declara en sus dependencias: `Requires-Dist` de su METADATA). Se usa el modelo `nmp.onnx` del repo, sin TensorFlow. — leído / medido (es lo que se instaló aquí)
2. **Preparar la partitura**: MusicXML con un atributo `id` único en cada `<note>`, **los mismos ids que usa el visor**. partitura toma el id de ese atributo (`partitura/io/importmusicxml.py:1335-1339`); si no existe (por ejemplo, un MusicXML exportado por music21), todas las notas quedan con id `None`. En ese caso la receta pone `n0, n1, …` por orden y avisa, pero esos ids no serán los del visor. — leído y medido (music21 → partitura dio 225 notas con el mismo id `'None'`)
3. **Preparar el audio**: cualquier formato que lea `soundfile`, con **sólo la obra** (o el movimiento) que corresponde a la partitura. El alineador de audio estima dónde empieza y acaba la música, pero no se ha probado con grabaciones que contengan otras obras. — SIN MEDIR
4. **Decidir las repeticiones**: `--repeticiones maximo` (todas) o `minimo` (ninguna). Si el intérprete hace otra cosa (sólo la primera repetición, *da capo* sin repetir, cortes), hay que editar la partitura antes. En las mediciones de arriba la estructura se tomó de (n)ASAP, es decir, se dio por conocida. — leído
5. **Ejecutar**:
   ```bash
   python lab_mapa_humano/receta_mapa.py grabacion.wav obra.musicxml mapa.json
   ```
   En este entorno (4 núcleos, CPU) tardó 22,9 s para 48 s de audio (Bach BWV 860), con la máquina cargada con otros procesos. — medido
6. **Salida** (`mapa.json`):
   ```json
   {"notas":   [{"id": "n1", "repeticion": 1, "ms": 998, "evidencia": "directa", "revisar": true}, ...],
    "compases": [{"compas": "1", "orden": 0, "ms": 998, "revisar": true}, ...]}
   ```
   `repeticion` separa el sufijo `-1`, `-2` que partitura añade al desplegar repeticiones; `evidencia` dice si el tiempo sale de una nota que basic-pitch oyó o de una interpolación. — leído (formato del script)
7. **Revisar** las notas y compases con `revisar: true` (ver 4.2) en el propio visor, escuchando.

Comprobación de punta a punta por la línea de comandos sobre Bach BWV 860 frente a la verdad de (n)ASAP: 602 notas, mediana |err| 9,2 ms, 92,2 % ≤ 50 ms, 2,3 % > 200 ms, 20,3 % marcadas. — medido

### 4.2 Qué necesita corrección humana
- **Antes de ejecutar** (manual, siempre): que los ids del MusicXML sean los del visor; la estructura de repeticiones y cortes; recortar el audio a la obra. — leído
- **Después** — medido sobre las 8 piezas de piano (`resultados/p05_marca_revisar.csv`):
  - La receta marca «revisar» cuando sus dos alineadores (A simbólico con basic-pitch y B de audio directo) discrepan más de 60 ms. Se marca el **20,8 % de las notas** (de 10,1 % en Mozart y la Fuga a 38,7 % en Debussy).
  - Esa marca contiene el **93,8 % de los errores > 100 ms** (793 de 845).
  - Entre las notas **no** marcadas, sólo el **0,6 %** tiene un error > 100 ms (52 de 8 520).
  - **Cuidado**: el umbral de 60 ms se eligió con estas mismas 8 piezas.
- **Dónde va a tocar corregir más** (apartado 3.3): obras con mucho pedal o resonancia (Debussy, Rajmáninov), los primeros compases, adornos y trinos, pasajes de rubato fuerte y, fuera del piano, casi todo (ver 3.4).
- **Precisión esperable sin tocar nada** (piano, 8 obras): mediana |err| 8,8 ms, 88,2 % de las notas a ≤ 50 ms, 4,5 % a > 200 ms; por compás, 89,7 % a ≤ 50 ms. El suelo del método es ±1 fotograma de basic-pitch (≈12 ms). — medido
- **Mejoras baratas no probadas** en la receta: `minimum_note_length` 58 ms (en 2.2 recupera la mitad de las notas perdidas) y corregir el diente de sierra del mapeo fotograma→tiempo (1.2). — SIN MEDIR dentro de la receta

## 5. Qué hay en la rama
Todo en `lab_mapa_humano/` (los datos y audios **no** se suben; se descargan con `bajar_datos.py` a un directorio de trabajo):

| fichero | qué hace |
|:--|:--|
| `comun.py` | inferencia ONNX con caché, decodificado con umbrales, evaluación con mir_eval, síntesis con fluidsynth |
| `seleccion_asap.csv` | las 8 interpretaciones de (n)ASAP/MAESTRO usadas |
| `bajar_datos.py` | baja de MAESTRO v3 sólo esos WAV/MIDI, leyendo el zip remoto por rangos HTTP |
| `p00_resolucion_temporal.py` | mapeo fotograma→tiempo y tonos sintéticos de onset exacto (1.2) |
| `p01_onsets_piano_maestro.py` | onsets en piano real y barrido de umbrales (2.1, 2.2) |
| `p02_onsets_sintetizado.py`, `p02b_tolerancia.py` | onsets por familia con audio sintetizado (2.3) |
| `p03_mapa_partitura.py` | mapa partitura→ms con los métodos A, B, C y T (3) |
| `p04_repetidas_y_melodia.py` | notas repetidas y `melodia_trick` (2.2, 2.4) |
| `p05_combinaciones.py` | combinación E, desgloses y calidad de la marca «revisar» (3, 4.2) |
| `p06_reverberacion.py` | reverberación añadida (3.3) |
| `p07_mapa_no_piano.py` | mapa partitura→ms con cuerda y conjunto sintetizados (3.4) |
| `p08_tipos_de_error.py` | clasificación de perdidas e inventadas (2.1, 2.3) |
| `receta_mapa.py` | la receta (4) |
| `navegador/bench.html`, `navegador/bench.js` | medición de ONNX y TF.js en Chromium (2.5) |
| `resultados/` | todas las tablas citadas (CSV/JSON) |

Orden para reproducir: `bajar_datos.py` → `p01` (corta los tramos y cachea el modelo) → `p03` → `p05`; el resto es independiente.

## SIN MEDIR

Cosas que este laboratorio **no** ha medido; cualquier afirmación sobre ellas es una suposición:

1. **Grabaciones humanas reales que no sean de piano** (cuerda, viento, voz, conjunto, orquesta). Sólo se probó audio sintetizado con FluidR3, porque Zenodo, Hugging Face, archive.org y el servidor de Vienna 4x22 estaban bloqueados por la red de este entorno (URMP, Bach10, GuitarSet, MusicNet, Dagstuhl ChoirSet…).
2. **Voz cantada**, con o sin acompañamiento.
3. Que la combinación E y el umbral de 60 ms de la marca «revisar» funcionen igual de bien en **obras distintas** de las 8 con las que se eligieron.
4. La receta con **`minimum_note_length` = 58 ms** o con el **mapeo exacto fotograma→tiempo** (los dos se midieron sólo para onsets, no dentro del mapa).
5. Grabaciones con **varias obras o movimientos**, silencios largos, aplausos, afinación, toses; grabaciones de **móvil o en mp3**; **sala real** con reverberación (aquí sólo una respuesta al impulso sintética).
6. Interpretaciones con **repeticiones, cortes o *da capo*** distintos de la partitura; detectar la estructura de repeticiones automáticamente.
7. **Afinación no estándar** (barroco a 415 Hz, etc.): basic-pitch redondea a semitono y no se probó.
8. **Adornos** con un número de casos suficiente (aquí n = 44 apoyaturas y 35 notas con ornamento).
9. Diferencias numéricas entre el modelo **TensorFlow original** y el **ONNX** usado aquí (TF no estaba instalado).
10. El **decodificado de basic-pitch-ts** (TypeScript): no se ejecutó; sólo se leyó y se midió la inferencia del modelo TF.js.
11. En navegador: tiempo del decodificado a notas y del alineado (parangonar no existe en JavaScript), memoria, móviles, **WebGL con GPU real** (el Chromium headless de aquí usa SwiftShader, por software), el backend `wasm` de TF.js y ONNX con varios hilos (necesita cabeceras COOP/COEP).
12. **Tiempo de la receta en una máquina descargada** (los 22,9 s se midieron con otros procesos corriendo).
13. Que los ids del MusicXML coincidan con los que use el visor (por ejemplo, si genera los suyos con Verovio).
14. Si las licencias no comerciales de algunos datos de entrenamiento (apartado 0) afectan al uso de los pesos: es una cuestión legal que el repo no aclara.
