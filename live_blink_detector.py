"""
live_blink_detector.py
----------------------
Detección de parpadeos en tiempo real usando la webcam.
Imprime por consola cada vez que se detecta un parpadeo y guarda dos CSV
al finalizar la sesión (equivalentes a los de blink_detector.py).

Uso:
    python3 live_blink_detector.py

Controles:
    Q  →  salir
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import pandas as pd
import subprocess
import time
import os
from datetime import datetime

# ---------------------------------------------------------------------------
# Landmarks de cada ojo (mismo orden que blink_detector.py)
# ---------------------------------------------------------------------------
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]

# ---------------------------------------------------------------------------
# Parámetros configurables
# ---------------------------------------------------------------------------
MODEL_PATH       = "face_landmarker.task"   # ya descargado por blink_detector.py
EAR_THRESHOLD    = 0.22
CONSEC_FRAMES    = 2

# Si se rellena, el script busca automáticamente una cámara cuyo nombre
# contenga este texto (sin distinguir mayúsculas).
# Ejemplos: "iphone", "continuity", "macbook"
# Déjalo vacío ("") para que pregunte siempre.
CAMERA_NAME_HINT = "iphone"
OUTPUT_DIR       = "resultados"


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calculate_ear(landmarks, eye_indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    p1, p2, p3, p4, p5, p6 = pts
    return (euclidean(p2, p6) + euclidean(p3, p5)) / (2.0 * euclidean(p1, p4))


# ---------------------------------------------------------------------------
# Selección de cámara
# ---------------------------------------------------------------------------

def get_camera_names():
    """
    Obtiene los nombres reales de las cámaras usando AVFoundation vía Swift.
    Devuelve una lista ordenada de nombres (índice 0 = cámara 0 de OpenCV).
    Si Swift no está disponible, devuelve lista vacía.
    """
    swift_code = (
        "import AVFoundation; "
        "AVCaptureDevice.DiscoverySession("
        "deviceTypes: [.builtInWideAngleCamera, .external, .continuityCamera], "
        "mediaType: .video, position: .unspecified"
        ").devices.forEach { print($0.localizedName) }"
    )
    try:
        result = subprocess.run(
            ["swift", "-e", swift_code],
            capture_output=True, text=True, timeout=10
        )
        names = [l.strip() for l in result.stdout.splitlines() if l.strip()]
        return names
    except Exception:
        return []


def list_cameras_with_names(max_test=8):
    """
    Devuelve lista de (índice_opencv, nombre) para cada cámara disponible.
    """
    names   = get_camera_names()
    indices = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            indices.append(i)
            cap.release()

    result = []
    for pos, idx in enumerate(indices):
        nombre = names[pos] if pos < len(names) else f"Cámara {idx}"
        result.append((idx, nombre))
    return result


def select_camera():
    """
    Si CAMERA_NAME_HINT está definido, busca automáticamente la cámara
    cuyo nombre lo contenga. Si no se encuentra o el hint está vacío,
    muestra la lista y pide elección manual.
    """
    print("Buscando cámaras disponibles...")
    cameras = list_cameras_with_names()

    if not cameras:
        print("ERROR: No se encontró ninguna cámara.")
        print("Si usas el iPhone, asegúrate de que esté desbloqueado y cerca.")
        raise SystemExit(1)

    # Intento de autoselección por nombre
    if CAMERA_NAME_HINT:
        hint = CAMERA_NAME_HINT.lower()
        for idx, nombre in cameras:
            if hint in nombre.lower():
                print(f"Cámara seleccionada automáticamente: [{idx}] {nombre}")
                return idx
        print(f"  (No se encontró ninguna cámara con '{CAMERA_NAME_HINT}' en el nombre.)")
        print(f"  Asegúrate de que el iPhone esté desbloqueado y cerca.\n")

    # Selección manual
    print("\nCámaras disponibles:")
    for idx, nombre in cameras:
        print(f"  [{idx}]  {nombre}")

    indices = [idx for idx, _ in cameras]
    print()
    while True:
        try:
            eleccion = int(input(f"Elige el número de cámara {indices}: "))
            if eleccion in indices:
                return eleccion
            print(f"  Opción no válida. Elige entre {indices}.")
        except ValueError:
            print("  Introduce un número.")


# ---------------------------------------------------------------------------
# Bucle principal
# ---------------------------------------------------------------------------

def run():
    camera_index = select_camera()
    print()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("ERROR: No se puede abrir la webcam.")
        raise SystemExit(1)

    # Usar modo VIDEO para que MediaPipe aproveche la continuidad temporal
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = mp_vision.FaceLandmarker.create_from_options(options)

    blink_counter  = 0
    consec_below   = 0
    in_blink       = False
    start_time_ms  = None
    ear_left_blink = []
    ear_right_blink= []

    blink_records = []   # un dict por parpadeo
    raw_records   = []   # un dict por fotograma

    print("Detección en vivo iniciada. Pulsa Q para salir.\n")

    session_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        timestamp_ms = int((time.time() - session_start) * 1000)

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = detector.detect_for_video(mp_image, timestamp_ms)

        ear_left      = None
        ear_right     = None
        ear_avg       = None
        face_detected = False
        ojo_cerrado   = False

        if result.face_landmarks:
            face_detected = True
            lms       = result.face_landmarks[0]
            ear_left  = calculate_ear(lms, LEFT_EYE_IDX,  w, h)
            ear_right = calculate_ear(lms, RIGHT_EYE_IDX, w, h)
            ear_avg   = (ear_left + ear_right) / 2.0
            ojo_cerrado = ear_avg < EAR_THRESHOLD

            if ojo_cerrado:
                consec_below += 1
                if not in_blink and consec_below >= CONSEC_FRAMES:
                    in_blink       = True
                    start_time_ms  = timestamp_ms
                    ear_left_blink = []
                    ear_right_blink= []
                if in_blink:
                    ear_left_blink.append(ear_left)
                    ear_right_blink.append(ear_right)
            else:
                if in_blink:
                    blink_counter += 1
                    duration_ms = timestamp_ms - start_time_ms
                    elapsed_s   = timestamp_ms / 1000
                    print(f"  [{elapsed_s:6.1f}s]  Parpadeo #{blink_counter:3d}  —  "
                          f"duración: {duration_ms} ms")
                    blink_records.append({
                        "parpadeo_id":         blink_counter,
                        "tiempo_inicio_ms":    start_time_ms,
                        "tiempo_fin_ms":       timestamp_ms,
                        "duracion_ms":         duration_ms,
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

        raw_records.append({
            "tiempo_ms":      timestamp_ms,
            "cara_detectada": face_detected,
            "ear_izquierdo":  round(ear_left,  4) if ear_left  is not None else None,
            "ear_derecho":    round(ear_right, 4) if ear_right is not None else None,
            "ear_promedio":   round(ear_avg,   4) if ear_avg   is not None else None,
            "ojo_cerrado":    ojo_cerrado,
            "en_parpadeo":    in_blink,
        })

        # --- Overlay en la ventana de previsualización ---
        color_estado = (0, 200, 0) if (face_detected and not in_blink) else \
                       (0, 100, 255) if in_blink else \
                       (80, 80, 80)

        estado_txt = "PARPADEO" if in_blink else ("OK" if face_detected else "sin cara")
        ear_txt    = f"EAR: {ear_avg:.3f}" if ear_avg is not None else "EAR: ---"

        cv2.putText(frame, estado_txt,                    (20, 40),  cv2.FONT_HERSHEY_SIMPLEX, 1.1, color_estado, 2)
        cv2.putText(frame, ear_txt,                       (20, 80),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        cv2.putText(frame, f"Parpadeos: {blink_counter}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

        cv2.imshow("Live Blink Detector  [Q = salir]", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # --- Cierre limpio ---
    detector.close()
    cap.release()
    cv2.destroyAllWindows()

    elapsed_total = time.time() - session_start
    bpm = (blink_counter / elapsed_total) * 60 if elapsed_total > 0 else 0

    # --- Guardar CSV ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sesion_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    blinks_path = os.path.join(OUTPUT_DIR, f"live_{sesion_id}_parpadeos.csv")
    raw_path    = os.path.join(OUTPUT_DIR, f"live_{sesion_id}_ear_raw.csv")

    pd.DataFrame(blink_records).to_csv(blinks_path, index=False, encoding="utf-8")
    pd.DataFrame(raw_records).to_csv(raw_path,       index=False, encoding="utf-8")

    print(f"\n--- Sesión terminada ---")
    print(f"Duración        : {elapsed_total:.1f} s")
    print(f"Total parpadeos : {blink_counter}")
    print(f"Frecuencia      : {bpm:.1f} parpadeos/min")
    print(f"\nArchivos generados:")
    print(f"  {blinks_path}")
    print(f"  {raw_path}")


if __name__ == "__main__":
    run()
