"""
blink_detector.py
-----------------
Detecta y registra cada parpadeo en un video de primer plano facial.
Usa MediaPipe Face Landmarker (Tasks API ≥0.10) para calcular el Eye Aspect
Ratio (EAR) en cada fotograma.

Uso:
    python blink_detector.py

El video debe estar en:   videos/VIDTOK1.<ext>   (mp4, avi, mov, mkv o wmv)
Los resultados se guardan en:
    resultados/VIDTOK1_parpadeos.csv   — un registro por parpadeo
    resultados/VIDTOK1_ear_raw.csv     — EAR fotograma a fotograma
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import pandas as pd
import os
import glob
import urllib.request

# ---------------------------------------------------------------------------
# Modelo Face Landmarker (se descarga automáticamente si no existe)
# ---------------------------------------------------------------------------
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = "face_landmarker.task"


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Descargando modelo Face Landmarker (~28 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Modelo descargado.\n")

# ---------------------------------------------------------------------------
# Índices de landmarks de MediaPipe Face Mesh para cada ojo
# Orden: [extremo_izq, sup_izq, sup_der, extremo_der, inf_der, inf_izq]
# ---------------------------------------------------------------------------
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]

# ---------------------------------------------------------------------------
# Parámetros configurables
# ---------------------------------------------------------------------------
VIDEO_DIR      = "videos"
VIDEO_NAME     = "VIDTOK1"
OUTPUT_DIR     = "resultados"

EAR_THRESHOLD  = 0.22   # EAR por debajo de este valor → ojo cerrado
CONSEC_FRAMES  = 2      # Mínimo de fotogramas consecutivos cerrados para contar un parpadeo


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calculate_ear(landmarks, eye_indices, frame_w, frame_h):
    """
    Calcula el Eye Aspect Ratio (EAR) a partir de los landmarks del ojo.

    Fórmula:
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    donde los puntos siguen el orden estándar de 6-puntos para cada ojo.
    Un valor cercano a 0 indica ojo completamente cerrado.
    """
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((lm.x * frame_w, lm.y * frame_h))

    p1, p2, p3, p4, p5, p6 = pts

    vertical_a = euclidean(p2, p6)
    vertical_b = euclidean(p3, p5)
    horizontal  = euclidean(p1, p4)

    ear = (vertical_a + vertical_b) / (2.0 * horizontal)
    return ear


def find_video(video_dir, video_name):
    """
    Busca el archivo de vídeo con cualquier extensión soportada.
    Devuelve la ruta completa o None si no lo encuentra.
    """
    extensions = ["mp4", "MP4", "avi", "AVI", "mov", "MOV", "mkv", "MKV", "wmv", "WMV"]
    for ext in extensions:
        path = os.path.join(video_dir, f"{video_name}.{ext}")
        if os.path.exists(path):
            return path
    # Búsqueda por glob como fallback
    matches = glob.glob(os.path.join(video_dir, f"{video_name}.*"))
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Función principal de detección
# ---------------------------------------------------------------------------

def detect_blinks(video_path, output_dir,
                  ear_threshold=EAR_THRESHOLD,
                  consec_frames=CONSEC_FRAMES):
    """
    Procesa el vídeo fotograma a fotograma y detecta parpadeos.

    Parámetros
    ----------
    video_path    : str   — ruta al archivo de vídeo
    output_dir    : str   — carpeta donde se guardarán los CSV
    ear_threshold : float — umbral EAR bajo el cual se considera ojo cerrado
    consec_frames : int   — número mínimo de fotogramas consecutivos cerrados
                            para registrar un parpadeo (filtra micro-ruidos)

    Devuelve
    --------
    blinks_df : pd.DataFrame — tabla con un registro por parpadeo
    raw_df    : pd.DataFrame — tabla con EAR fotograma a fotograma
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se puede abrir el vídeo: {video_path}")

    fps           = cap.get(cv2.CAP_PROP_FPS)
    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s    = total_frames / fps if fps > 0 else 0

    print(f"\nVídeo cargado : {video_path}")
    print(f"Resolución    : {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} px")
    print(f"FPS           : {fps:.2f}")
    print(f"Total frames  : {total_frames}")
    print(f"Duración      : {duration_s:.2f} s")
    print(f"Umbral EAR    : {ear_threshold}")
    print(f"Frames mínimos: {consec_frames}")
    print("\nProcesando...\n")

    raw_records   = []   # Un dict por fotograma
    blink_records = []   # Un dict por parpadeo

    blink_counter      = 0
    consec_below       = 0
    in_blink           = False
    blink_start_frame  = None
    blink_start_time   = None
    ear_left_blink     = []   # EAR ojo izquierdo durante el parpadeo
    ear_right_blink    = []   # EAR ojo derecho durante el parpadeo

    ensure_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = mp_vision.FaceLandmarker.create_from_options(options)

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_time_ms = (frame_idx / fps) * 1000 if fps > 0 else 0
        h, w = frame.shape[:2]

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = detector.detect(mp_image)

        ear_left      = None
        ear_right     = None
        ear_avg       = None
        face_detected = False
        ojo_cerrado   = False

        if result.face_landmarks:
            face_detected = True
            lms = result.face_landmarks[0]

            ear_left    = calculate_ear(lms, LEFT_EYE_IDX,  w, h)
            ear_right   = calculate_ear(lms, RIGHT_EYE_IDX, w, h)
            ear_avg     = (ear_left + ear_right) / 2.0
            ojo_cerrado = ear_avg < ear_threshold

            # --- Máquina de estados del parpadeo ---
            if ojo_cerrado:
                consec_below += 1

                # Marcar inicio del parpadeo al cumplir el mínimo de frames
                if not in_blink and consec_below >= consec_frames:
                    in_blink = True
                    blink_start_frame = frame_idx - consec_below + 1
                    blink_start_time  = (blink_start_frame / fps) * 1000
                    ear_left_blink    = []
                    ear_right_blink   = []

                if in_blink:
                    ear_left_blink.append(ear_left)
                    ear_right_blink.append(ear_right)

            else:
                if in_blink:
                    # El parpadeo ha terminado → cerrar registro
                    blink_counter += 1
                    blink_end_frame = frame_idx - 1
                    blink_end_time  = (blink_end_frame / fps) * 1000
                    duration_ms     = blink_end_time - blink_start_time

                    blink_records.append({
                        "parpadeo_id":         blink_counter,
                        "frame_inicio":        blink_start_frame,
                        "frame_fin":           blink_end_frame,
                        "tiempo_inicio_ms":    round(blink_start_time, 2),
                        "tiempo_fin_ms":       round(blink_end_time, 2),
                        "duracion_ms":         round(duration_ms, 2),
                        "n_frames_cerrado":    len(ear_left_blink),
                        "ear_min_izquierdo":   round(min(ear_left_blink),  4) if ear_left_blink  else None,
                        "ear_min_derecho":     round(min(ear_right_blink), 4) if ear_right_blink else None,
                        "ear_min_promedio":    round(
                            min((l + r) / 2 for l, r in zip(ear_left_blink, ear_right_blink)), 4
                        ) if ear_left_blink else None,
                        "ear_media_izquierdo": round(np.mean(ear_left_blink),  4) if ear_left_blink  else None,
                        "ear_media_derecho":   round(np.mean(ear_right_blink), 4) if ear_right_blink else None,
                    })

                in_blink        = False
                consec_below    = 0
                ear_left_blink  = []
                ear_right_blink = []

        # --- Registro fotograma a fotograma ---
        raw_records.append({
            "frame":          frame_idx,
            "tiempo_ms":      round(frame_time_ms, 2),
            "cara_detectada": face_detected,
            "ear_izquierdo":  round(ear_left,  4) if ear_left  is not None else None,
            "ear_derecho":    round(ear_right, 4) if ear_right is not None else None,
            "ear_promedio":   round(ear_avg,   4) if ear_avg   is not None else None,
            "ojo_cerrado":    ojo_cerrado,
            "en_parpadeo":    in_blink,
        })

        frame_idx += 1
        if frame_idx % 150 == 0:
            pct = 100 * frame_idx / total_frames if total_frames else 0
            print(f"  {frame_idx}/{total_frames} frames ({pct:.1f}%) — "
                  f"parpadeos hasta ahora: {blink_counter}")

    detector.close()

    # Parpadeo todavía abierto al acabar el vídeo → cerrar igualmente
    if in_blink and ear_left_blink:
        blink_counter += 1
        blink_end_frame = frame_idx - 1
        blink_end_time  = (blink_end_frame / fps) * 1000
        duration_ms     = blink_end_time - blink_start_time

        blink_records.append({
            "parpadeo_id":         blink_counter,
            "frame_inicio":        blink_start_frame,
            "frame_fin":           blink_end_frame,
            "tiempo_inicio_ms":    round(blink_start_time, 2),
            "tiempo_fin_ms":       round(blink_end_time, 2),
            "duracion_ms":         round(duration_ms, 2),
            "n_frames_cerrado":    len(ear_left_blink),
            "ear_min_izquierdo":   round(min(ear_left_blink),  4) if ear_left_blink  else None,
            "ear_min_derecho":     round(min(ear_right_blink), 4) if ear_right_blink else None,
            "ear_min_promedio":    round(
                min((l + r) / 2 for l, r in zip(ear_left_blink, ear_right_blink)), 4
            ) if ear_left_blink else None,
            "ear_media_izquierdo": round(np.mean(ear_left_blink),  4) if ear_left_blink  else None,
            "ear_media_derecho":   round(np.mean(ear_right_blink), 4) if ear_right_blink else None,
        })

    cap.release()

    # --- Guardar CSV ---
    video_stem = os.path.splitext(os.path.basename(video_path))[0]

    raw_df    = pd.DataFrame(raw_records)
    blinks_df = pd.DataFrame(blink_records)

    raw_path    = os.path.join(output_dir, f"{video_stem}_ear_raw.csv")
    blinks_path = os.path.join(output_dir, f"{video_stem}_parpadeos.csv")

    raw_df.to_csv(raw_path,    index=False, encoding="utf-8")
    blinks_df.to_csv(blinks_path, index=False, encoding="utf-8")

    # --- Resumen en consola ---
    bpm = (blink_counter / duration_s) * 60 if duration_s > 0 else 0

    print(f"\n{'='*50}")
    print(f"  RESUMEN")
    print(f"{'='*50}")
    print(f"  Total parpadeos detectados : {blink_counter}")
    print(f"  Duración del vídeo         : {duration_s:.2f} s")
    print(f"  Frecuencia de parpadeo     : {bpm:.1f} parpadeos/min")
    if not blinks_df.empty:
        print(f"  Duración media parpadeo    : {blinks_df['duracion_ms'].mean():.1f} ms")
        print(f"  Duración mínima parpadeo   : {blinks_df['duracion_ms'].min():.1f} ms")
        print(f"  Duración máxima parpadeo   : {blinks_df['duracion_ms'].max():.1f} ms")
    print(f"{'='*50}")
    print(f"\nArchivos generados:")
    print(f"  {blinks_path}  ← registro de parpadeos")
    print(f"  {raw_path}     ← EAR fotograma a fotograma")

    return blinks_df, raw_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    video_path = find_video(VIDEO_DIR, VIDEO_NAME)

    if video_path is None:
        print(f"\nERROR: No se encontró '{VIDEO_NAME}' en la carpeta '{VIDEO_DIR}'.")
        print("Formatos soportados: mp4, avi, mov, mkv, wmv")
        raise SystemExit(1)

    detect_blinks(
        video_path=video_path,
        output_dir=OUTPUT_DIR,
        ear_threshold=EAR_THRESHOLD,
        consec_frames=CONSEC_FRAMES,
    )
