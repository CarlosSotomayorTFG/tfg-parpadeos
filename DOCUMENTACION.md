# Detector de parpadeos — Documentación técnica

## Objetivo

Este script analiza un vídeo de primer plano facial y extrae el registro bruto de cada parpadeo detectado. El propósito final es medir el impacto de los vídeos cortos (TikTok, Reels, etc.) en la **frecuencia de parpadeo** del espectador.

---

## Estructura del proyecto

```
Primera Ojos/
├── blink_detector.py       ← script principal
├── requirements.txt        ← dependencias Python
├── DOCUMENTACION.md        ← este archivo
├── videos/
│   └── VIDTOK1.<ext>       ← vídeo a analizar (mp4, avi, mov, mkv, wmv)
└── resultados/
    ├── VIDTOK1_parpadeos.csv   ← generado al ejecutar
    └── VIDTOK1_ear_raw.csv     ← generado al ejecutar
```

---

## Instalación

### Requisitos previos
- Python 3.9 o superior
- pip

### Pasos

```bash
# 1. (Recomendado) Crear entorno virtual
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 2. Instalar dependencias
pip install -r requirements.txt
```

---

## Uso

1. Coloca el vídeo en la carpeta `videos/` con el nombre `VIDTOK1` (cualquier extensión soportada).
2. Ejecuta el script:

```bash
python blink_detector.py
```

3. Los resultados aparecen en la carpeta `resultados/`.

---

## Fundamento técnico

### MediaPipe Face Mesh

El script usa **MediaPipe Face Mesh** de Google, que localiza **468 landmarks faciales** en tiempo real. De esos 468 puntos, se utilizan 12 (6 por ojo) para calcular el estado de apertura ocular.

```
Landmarks del ojo izquierdo  : [362, 385, 387, 263, 373, 380]
Landmarks del ojo derecho    : [33,  160, 158, 133, 153, 144]
```

### Eye Aspect Ratio (EAR)

El **EAR** es una métrica geométrica propuesta por Soukupová & Čech (2016) que cuantifica el grado de apertura del ojo a partir de 6 puntos:

```
        p2    p3
p1  ·         ·  p4
        p6    p5

EAR = (||p2−p6|| + ||p3−p5||) / (2 × ||p1−p4||)
```

| EAR aproximado | Estado |
|---|---|
| 0.25 – 0.35 | Ojo completamente abierto |
| 0.15 – 0.22 | Ojo cerrándose / parpadeo en curso |
| < 0.10 | Ojo completamente cerrado |

El EAR se calcula por separado para cada ojo. El script trabaja sobre el **promedio de ambos** para decidir si hay parpadeo.

### Algoritmo de detección

```
Para cada fotograma:
  1. Calcular EAR_izq y EAR_der
  2. ear_promedio = (EAR_izq + EAR_der) / 2
  3. Si ear_promedio < EAR_THRESHOLD:
       incrementar contador_frames_cerrado
       Si contador_frames_cerrado >= CONSEC_FRAMES y no estamos en parpadeo:
           → INICIO del parpadeo (registrar frame y timestamp)
  4. Si ear_promedio >= EAR_THRESHOLD y estábamos en parpadeo:
       → FIN del parpadeo (registrar duración y métricas EAR)
       reiniciar estado
```

El parámetro `CONSEC_FRAMES` actúa como **filtro anti-ruido**: evita contar como parpadeo una bajada de EAR de un solo fotograma causada por compresión de vídeo o movimiento brusco.

---

## Parámetros configurables

Los parámetros se ajustan directamente al inicio de `blink_detector.py`:

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `VIDEO_DIR` | `"videos"` | Carpeta donde está el vídeo |
| `VIDEO_NAME` | `"VIDTOK1"` | Nombre del vídeo sin extensión |
| `OUTPUT_DIR` | `"resultados"` | Carpeta de salida |
| `EAR_THRESHOLD` | `0.22` | Umbral EAR para considerar ojo cerrado |
| `CONSEC_FRAMES` | `2` | Frames mínimos cerrados para registrar parpadeo |

### Ajuste del umbral EAR

El umbral óptimo puede variar según la persona y las condiciones del vídeo:

- **Subirlo (0.25)** si se pierden parpadeos reales (falsos negativos).
- **Bajarlo (0.18)** si se detectan parpadeos que no existen (falsos positivos).
- Como referencia: en condiciones normales el EAR de un ojo abierto está entre 0.25 y 0.35.

---

## Archivos de salida

### `VIDTOK1_parpadeos.csv` — registro bruto de parpadeos

Un registro por parpadeo detectado. Este es el fichero principal para el análisis.

| Columna | Tipo | Descripción |
|---|---|---|
| `parpadeo_id` | int | Número de parpadeo en orden cronológico |
| `frame_inicio` | int | Fotograma en que empieza el parpadeo |
| `frame_fin` | int | Fotograma en que termina el parpadeo |
| `tiempo_inicio_ms` | float | Timestamp de inicio en milisegundos |
| `tiempo_fin_ms` | float | Timestamp de fin en milisegundos |
| `duracion_ms` | float | Duración total del parpadeo en ms |
| `n_frames_cerrado` | int | Número de fotogramas con ojo cerrado |
| `ear_min_izquierdo` | float | EAR mínimo del ojo izquierdo durante el parpadeo |
| `ear_min_derecho` | float | EAR mínimo del ojo derecho durante el parpadeo |
| `ear_min_promedio` | float | EAR mínimo promedio (cierre máximo alcanzado) |
| `ear_media_izquierdo` | float | EAR medio del ojo izquierdo durante el parpadeo |
| `ear_media_derecho` | float | EAR medio del ojo derecho durante el parpadeo |

**Ejemplo de fila:**

```
parpadeo_id, frame_inicio, frame_fin, tiempo_inicio_ms, tiempo_fin_ms, duracion_ms, n_frames_cerrado, ear_min_izquierdo, ...
1,           142,          145,       4733.33,           4833.33,       100.0,       3,                0.1023, ...
```

### `VIDTOK1_ear_raw.csv` — EAR fotograma a fotograma

Registro completo de cada fotograma procesado. Útil para visualizar la curva de EAR y auditar la detección.

| Columna | Tipo | Descripción |
|---|---|---|
| `frame` | int | Índice del fotograma (0-based) |
| `tiempo_ms` | float | Timestamp en milisegundos |
| `cara_detectada` | bool | Si MediaPipe localizó una cara |
| `ear_izquierdo` | float | EAR ojo izquierdo (null si no hay cara) |
| `ear_derecho` | float | EAR ojo derecho (null si no hay cara) |
| `ear_promedio` | float | Promedio de ambos ojos |
| `ojo_cerrado` | bool | Si ear_promedio < EAR_THRESHOLD |
| `en_parpadeo` | bool | Si ese fotograma pertenece a un parpadeo registrado |

---

## Métricas derivadas útiles para el análisis

A partir de `_parpadeos.csv` se pueden calcular directamente:

```python
import pandas as pd

df = pd.read_csv("resultados/VIDTOK1_parpadeos.csv")

# Frecuencia de parpadeo (parpadeos por minuto)
duracion_video_min = ...   # en minutos
bpm = len(df) / duracion_video_min

# Duración media de parpadeo
media_ms = df["duracion_ms"].mean()

# Distribución temporal (inter-blink interval)
df["ibi_ms"] = df["tiempo_inicio_ms"].diff()   # intervalo entre parpadeos
```

---

## Limitaciones conocidas

| Situación | Efecto | Mitigación |
|---|---|---|
| Cara parcialmente ocluida | MediaPipe falla → frames sin EAR | Asegurar plano frontal despejado |
| Vídeo de baja resolución o muy comprimido | EAR ruidoso → falsos positivos | Subir `CONSEC_FRAMES` a 3 |
| Gafas con montura gruesa | Landmarks desplazados | Ajustar `EAR_THRESHOLD` manualmente |
| Luz directa frontal muy intensa | Saturación → landmarks inestables | Mejorar iluminación del vídeo |
| Parpadeos extremadamente rápidos | < `CONSEC_FRAMES` → no detectados | Bajar `CONSEC_FRAMES` a 1 (más ruido) |

---

## Dependencias

| Librería | Versión mínima | Uso |
|---|---|---|
| `mediapipe` | 0.10.0 | Localización de landmarks faciales |
| `opencv-python` | 4.8.0 | Decodificación de vídeo |
| `numpy` | 1.24.0 | Cálculos vectoriales (distancias euclídeas) |
| `pandas` | 2.0.0 | Creación y exportación de los CSV |

---

## Referencias

- Soukupová, T., & Čech, J. (2016). *Real-Time Eye Blink Detection using Facial Landmarks*. CVWW 2016.
- MediaPipe Face Mesh: https://google.github.io/mediapipe/solutions/face_mesh
