# Explicación técnica del proyecto — Detector de parpadeos

> Este documento explica **qué hace el código, por qué funciona así y qué decisiones se tomaron**,
> con el nivel de detalle necesario para comprenderlo y defenderlo.

---

## 1. Qué problema resuelve este proyecto

El objetivo del TFG es medir si los vídeos cortos (TikTok, Reels...) afectan a la frecuencia de parpadeo de quien los ve. Para eso necesitamos:

1. Grabar a una persona viendo un vídeo.
2. Detectar automáticamente cada vez que parpadea.
3. Registrar el momento exacto y la duración de cada parpadeo.

El resultado es un fichero de datos que luego se puede analizar estadísticamente.

---

## 2. Herramientas utilizadas

### Python
Lenguaje de programación elegido por su ecosistema científico. Todo el código está escrito en Python 3.

### OpenCV (`cv2`)
Librería de visión por computador. En este proyecto se usa para **leer los vídeos fotograma a fotograma** (o capturar imagen de la webcam en tiempo real). Piénsalo como el "reproductor de vídeo" del código: abre el archivo, extrae cada imagen y nos la entrega para procesarla.

### MediaPipe (de Google)
Librería de inteligencia artificial que localiza puntos faciales en una imagen. En concreto usamos su modelo **Face Landmarker**, que detecta **468 puntos** (llamados *landmarks*) distribuidos por la cara: cejas, nariz, boca, contorno facial... y también los ojos, que es lo que nos interesa.

### NumPy
Librería matemática. La usamos para calcular distancias euclídeas entre puntos (la distancia en línea recta entre dos coordenadas).

### Pandas
Librería de tablas de datos. La usamos para construir los resultados y exportarlos a archivos `.csv` (que se pueden abrir en Excel).

---

## 3. El concepto clave: Eye Aspect Ratio (EAR)

### ¿Qué es?
El EAR (Ratio de Aspecto del Ojo) es una fórmula matemática que mide **cuánto de abierto está un ojo** usando solo 6 puntos de su contorno. Fue propuesta por Soukupová y Čech en 2016 y es el estándar en detección de parpadeos.

### Los 6 puntos del ojo
Para cada ojo se usan 6 puntos en posiciones concretas:

```
        p2      p3
   p1  ·    ·    ·  p4
        p6      p5
```

- **p1 y p4**: extremos izquierdo y derecho del ojo (la anchura)
- **p2, p3, p5, p6**: puntos superior e inferior (la altura)

### La fórmula

```
EAR = (distancia(p2,p6) + distancia(p3,p5)) / (2 × distancia(p1,p4))
```

En palabras: **la suma de las dos alturas verticales del ojo, dividida entre el doble de su anchura**.

### ¿Por qué funciona?

- Cuando el ojo está **abierto**: las alturas verticales son grandes → EAR ≈ 0.25 a 0.35
- Cuando el ojo se **cierra** (parpadeo): las alturas se reducen a casi cero → EAR cae por debajo de 0.22
- Cuando vuelve a **abrirse**: el EAR sube de nuevo

La clave es que dividir entre la anchura hace que el valor sea **independiente de la distancia a la cámara**: tanto si la persona está lejos como cerca, el EAR tiene el mismo rango de valores.

### ¿Por qué se promedian los dos ojos?
Usamos el promedio de EAR izquierdo y EAR derecho para mayor robustez. Si solo usáramos un ojo y ese ojo tuviera un reflejo de luz o quedara parcialmente ocluido, tendríamos un falso positivo. Con el promedio, un problema en un ojo solo afecta a la mitad de la señal.

---

## 4. Cómo funciona MediaPipe en este proyecto

### El modelo Face Landmarker
MediaPipe incluye un modelo de red neuronal preentrenado (el archivo `face_landmarker.task`, de 28 MB) que, dada una imagen, devuelve las coordenadas de 468 puntos faciales. Este modelo ha sido entrenado por Google con millones de imágenes y nosotros lo usamos directamente sin modificarlo.

### Coordenadas normalizadas
MediaPipe devuelve las coordenadas de los landmarks en formato **normalizado**: valores entre 0 y 1, donde (0,0) es la esquina superior izquierda de la imagen y (1,1) es la inferior derecha. Para convertirlas a píxeles reales multiplicamos por el ancho y alto del fotograma:

```python
x_pixels = landmark.x * ancho_frame
y_pixels = landmark.y * alto_frame
```

### Los índices de los ojos
De los 468 landmarks, nos interesan 12 (6 por ojo). Los índices concretos fueron elegidos de la documentación oficial de MediaPipe y corresponden a los puntos del contorno de cada ojo:

```python
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]  # ojo izquierdo
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]  # ojo derecho
```

El orden dentro de cada lista sigue el orden p1→p6 de la fórmula EAR.

---

## 5. El algoritmo de detección de parpadeos

Una vez que tenemos el EAR de cada fotograma, necesitamos detectar cuándo empieza y termina un parpadeo. Usamos una **máquina de estados** con dos estados: `en_parpadeo = False` y `en_parpadeo = True`.

### Umbral y filtro de ruido

Hay dos parámetros clave:
- **`EAR_THRESHOLD = 0.22`**: si el EAR baja de este valor, consideramos el ojo cerrado.
- **`CONSEC_FRAMES = 2`**: el ojo tiene que estar cerrado al menos 2 fotogramas seguidos para registrar un parpadeo.

El segundo parámetro existe porque los vídeos comprimidos o los movimientos bruscos pueden causar bajadas puntuales del EAR de un solo fotograma que no son parpadeos reales. Exigir 2 fotogramas consecutivos filtra ese ruido.

### La máquina de estados, paso a paso

```
Para cada fotograma del vídeo:

  1. Calcular EAR_izquierdo y EAR_derecho con los landmarks
  2. ear_promedio = (EAR_izq + EAR_der) / 2

  3. Si ear_promedio < 0.22:
       → incrementar contador de frames cerrados
       → Si el contador llega a 2 Y no estábamos ya en un parpadeo:
             INICIO del parpadeo: guardar frame actual y timestamp

  4. Si ear_promedio >= 0.22 Y estábamos en un parpadeo:
       → FIN del parpadeo: calcular duración, guardar registro
       → Reiniciar el estado (en_parpadeo = False, contador = 0)
```

### Por qué registrar el inicio cuando el contador llega a 2 y no antes
Cuando detectamos el segundo fotograma cerrado, sabemos que el cierre es real (no ruido). Pero el parpadeo empezó realmente en el fotograma anterior (el primero cerrado). Por eso el código hace:

```python
blink_start_frame = frame_idx - consec_below + 1
```

Esto "retrocede" al primer fotograma cerrado para que el timestamp de inicio sea correcto.

---

## 6. El script `blink_detector.py` — procesamiento de vídeo

Este script procesa un archivo de vídeo grabado previamente. Su flujo de ejecución es:

### 1. Buscar el vídeo
La función `find_video()` busca en la carpeta `videos/` un archivo llamado `VIDTOK1` con cualquier extensión (mp4, mov, avi...). Así el código no depende de que el usuario recuerde la extensión exacta.

### 2. Abrir el vídeo con OpenCV
```python
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
```
`cap` es el objeto que gestiona el vídeo. `fps` es la velocidad (fotogramas por segundo), necesaria para convertir números de fotograma en tiempos reales. Un vídeo a 30 fps con el parpadeo en el fotograma 300 → el parpadeo ocurrió a los 10 segundos.

### 3. Inicializar el detector de MediaPipe
```python
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.FaceLandmarkerOptions(...)
detector = mp_vision.FaceLandmarker.create_from_options(options)
```
Se carga el modelo de red neuronal. Esto ocurre una sola vez antes del bucle.

### 4. Bucle fotograma a fotograma
```python
while True:
    ret, frame = cap.read()   # leer siguiente fotograma
    if not ret:
        break                 # fin del vídeo
```
En cada iteración:
- Se convierte el fotograma de BGR (formato OpenCV) a RGB (formato MediaPipe)
- Se envía al detector: `result = detector.detect(mp_image)`
- Se calculan los EAR y se ejecuta la máquina de estados

### 5. Guardar los resultados
Al terminar el bucle se generan dos archivos CSV con `pandas`.

---

## 7. El script `live_blink_detector.py` — detección en tiempo real

Este script hace lo mismo pero con la cámara en vivo en lugar de un archivo. Las diferencias técnicas respecto al anterior son:

### Modo VIDEO vs IMAGE
El script de vídeo usa `detector.detect(imagen)` (modo IMAGE: cada fotograma es independiente).
El script en vivo usa `detector.detect_for_video(imagen, timestamp_ms)` (modo VIDEO: MediaPipe recibe también el timestamp y puede usar información temporal entre fotogramas para un tracking más estable).

### Timestamps en milisegundos
En el vídeo grabado el timestamp se calcula como `(numero_frame / fps) * 1000`. En el vivo se calcula con el reloj del sistema:
```python
session_start = time.time()
timestamp_ms = int((time.time() - session_start) * 1000)
```

### Selección de cámara
La función `get_camera_names()` lanza un comando Swift en segundo plano que consulta al sistema operativo (AVFoundation de macOS) los nombres reales de las cámaras disponibles. Luego los empareja con los índices de OpenCV (que son numéricos: 0, 1, 2...) por orden de aparición.

El parámetro `CAMERA_NAME_HINT = "iphone"` permite seleccionar automáticamente la cámara cuyo nombre contenga esa palabra, sin intervención manual.

### La Continuity Camera del iPhone
Cuando un iPhone está desbloqueado y cerca de un Mac (con el mismo Apple ID, Bluetooth y WiFi activos), macOS lo expone como una cámara virtual más. Se llama "Continuity Camera" y desde el punto de vista del código es indistinguible de cualquier otra cámara: OpenCV le asigna un índice y lee sus fotogramas exactamente igual.

---

## 8. Los archivos de salida

### `VIDTOK1_parpadeos.csv`
Una fila por parpadeo detectado. Es el archivo principal para el análisis estadístico. Columnas relevantes:

| Columna | Qué significa |
|---|---|
| `parpadeo_id` | Número de orden del parpadeo en el vídeo |
| `tiempo_inicio_ms` | Cuándo empezó (en milisegundos desde el inicio del vídeo) |
| `duracion_ms` | Cuánto duró el parpadeo (un parpadeo normal: 100-400 ms) |
| `ear_min_promedio` | El punto de mayor cierre del ojo durante el parpadeo |

### `VIDTOK1_ear_raw.csv`
Una fila por fotograma. Permite reconstruir la curva completa del EAR a lo largo del vídeo. Útil para visualizar y auditar: si un parpadeo parece sospechoso, se puede buscar en este archivo y ver exactamente qué valores de EAR tuvo.

---

## 9. Decisiones de diseño y sus alternativas

### ¿Por qué MediaPipe y no dlib o Haar Cascades?
- **Haar Cascades** (el detector clásico de OpenCV): rápido pero poco preciso, falla con caras ladeadas o con iluminación irregular.
- **dlib**: muy preciso, pero requiere descargar un modelo grande y tiene peor rendimiento en Mac con Apple Silicon.
- **MediaPipe**: ofrece los 468 landmarks con alta precisión, está optimizado para Apple Silicon (usa Metal GPU), y su modelo se descarga solo desde el código. Es el estado del arte accesible.

### ¿Por qué un umbral fijo (0.22) y no uno adaptativo?
Un umbral adaptativo (que aprenda el EAR base de cada persona) sería más robusto, pero introduce complejidad: necesita un período de calibración al inicio, y puede fallar si la persona parpadea durante esa fase. Para un primer plano frontal con buena iluminación, el umbral fijo de 0.22 funciona bien en la práctica.

### ¿Por qué CSV y no una base de datos?
El volumen de datos es pequeño (un vídeo de 90 segundos a 30 fps = 2700 filas en el raw). CSV es suficiente, se abre directamente en Excel o Python, y no requiere instalar ningún servidor de base de datos.

---

## 10. Estructura del proyecto y para qué sirve cada archivo

```
Primera Ojos/
│
├── blink_detector.py       → Procesa un vídeo grabado y genera los CSV
├── live_blink_detector.py  → Detección en tiempo real con cámara
├── requirements.txt        → Lista de librerías necesarias (para instalarlas con pip)
│
├── videos/
│   └── VIDTOK1.mov         → El vídeo a analizar (no se sube a GitHub)
│
├── resultados/             → Carpeta donde se guardan los CSV (no se sube a GitHub)
│
├── face_landmarker.task    → Modelo de IA descargado automáticamente (no se sube)
│
├── DOCUMENTACION.md        → Referencia técnica (parámetros, columnas, fórmulas)
└── EXPLICACION_TECNICA.md  → Este documento
```

El archivo `.gitignore` le dice a Git qué carpetas y archivos **no** subir a GitHub:
- `resultados/`: son datos locales del experimento, no código
- `face_landmarker.task`: pesa 28 MB y el código lo descarga solo si no existe
- `videos/`: los vídeos originales pueden contener imagen de personas (privacidad)

---

## 11. Flujo completo de una sesión de análisis

```
1. Grabar vídeo → guardarlo en videos/VIDTOK1.mov
2. Ejecutar:  python3 blink_detector.py
3. El script:
   a. Encuentra el vídeo en la carpeta videos/
   b. Descarga el modelo si no existe
   c. Lee cada fotograma con OpenCV
   d. Envía cada fotograma a MediaPipe → obtiene 468 landmarks
   e. Calcula EAR izquierdo y derecho con los 6 puntos de cada ojo
   f. Decide si hay parpadeo según umbral y frames consecutivos
   g. Registra inicio/fin de cada parpadeo
4. Genera dos CSV en resultados/
5. Analizar los CSV con Excel, Python o R
```

---

## 12. Glosario rápido

| Término | Definición |
|---|---|
| **Landmark** | Punto de referencia en una imagen (aquí: punto facial) |
| **EAR** | Eye Aspect Ratio — ratio numérico de apertura del ojo |
| **FPS** | Frames Per Second — fotogramas por segundo del vídeo |
| **Fotograma** | Imagen individual de un vídeo (como un fotograma de película) |
| **CSV** | Comma-Separated Values — tabla de datos en texto plano |
| **AVFoundation** | Framework de Apple para gestionar cámaras y audio en macOS |
| **Continuity Camera** | Función de Apple que permite usar el iPhone como webcam del Mac |
| **Red neuronal** | Modelo de IA entrenado con datos para resolver una tarea |
| **Umbral** | Valor límite a partir del cual se toma una decisión binaria |
| **Máquina de estados** | Sistema que solo puede estar en un estado a la vez y cambia entre ellos por eventos |
