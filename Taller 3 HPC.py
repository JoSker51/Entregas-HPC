import cv2
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# =====================================================
# CONFIGURACIÓN
# =====================================================
VIDEO_PATH = "C:/Users/santi/OneDrive/Escritorio/HOMEWORK/HPC/input.mp4"
OUTPUT_SEQ = "output_gray_seq.mp4"
OUTPUT_PAR = "output_gray_par.mp4"
FRAMES_DIR = "frames"
GRAY_SEQ_DIR = "gray_seq"
GRAY_PAR_DIR = "gray_par"
MAX_THREADS = os.cpu_count() or 4

# =====================================================
# FUNCIONES AUXILIARES
# =====================================================

def ensure_dir(path):
    """Crea una carpeta si no existe."""
    if not os.path.exists(path):
        os.makedirs(path)

def extract_frames(video_path, output_dir):
    """Extrae todos los frames del video y los guarda en output_dir."""
    ensure_dir(output_dir)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    cap.release()
    print(f"[INFO] {frame_count} frames extraídos en '{output_dir}'.")
    return frame_count

def process_frame_to_gray(input_path, output_path):
    """Convierte una imagen a escala de grises y la guarda."""
    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(output_path, gray)

def rebuild_video(frames_dir, output_video_path, fps=30):
    """Reconstruye un video a partir de los frames en frames_dir."""
    images = sorted(os.listdir(frames_dir))
    if not images:
        print("[ERROR] No hay frames para reconstruir el video.")
        return
    first_frame = cv2.imread(os.path.join(frames_dir, images[0]))
    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for img_name in images:
        frame = cv2.imread(os.path.join(frames_dir, img_name), cv2.IMREAD_GRAYSCALE)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"[INFO] Video generado correctamente: {output_video_path}")

# =====================================================
# ALGORITMO SECUENCIAL
# =====================================================

def sequential_gray(frames_dir, output_dir):
    ensure_dir(output_dir)
    frames = sorted(os.listdir(frames_dir))
    start = time.time()
    for frame_name in frames:
        input_path = os.path.join(frames_dir, frame_name)
        output_path = os.path.join(output_dir, frame_name)
        process_frame_to_gray(input_path, output_path)
    end = time.time()
    print(f"[SEQ] Tiempo total: {end - start:.2f} s")
    return end - start

# =====================================================
# ALGORITMO PARALELO
# =====================================================

def parallel_gray(frames_dir, output_dir, max_threads=4):
    ensure_dir(output_dir)
    frames = sorted(os.listdir(frames_dir))
    start = time.time()
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = []
        for frame_name in frames:
            input_path = os.path.join(frames_dir, frame_name)
            output_path = os.path.join(output_dir, frame_name)
            futures.append(executor.submit(process_frame_to_gray, input_path, output_path))
        for _ in as_completed(futures):
            pass
    end = time.time()
    print(f"[PAR] Tiempo total ({max_threads} hilos): {end - start:.2f} s")
    return end - start

# =====================================================
# PROGRAMA PRINCIPAL
# =====================================================

if __name__ == "__main__":
    print("=== Procesamiento de video a escala de grises ===\n")

    # 1. Extraer frames del video
    total_frames = extract_frames(VIDEO_PATH, FRAMES_DIR)

    # 2. Ejecutar versión secuencial
    t_seq = sequential_gray(FRAMES_DIR, GRAY_SEQ_DIR)

    # 3. Ejecutar versión paralela
    t_par = parallel_gray(FRAMES_DIR, GRAY_PAR_DIR, max_threads=MAX_THREADS)

    # 4. Reconstruir videos
    rebuild_video(GRAY_SEQ_DIR, OUTPUT_SEQ)
    rebuild_video(GRAY_PAR_DIR, OUTPUT_PAR)

    # 5. Comparar resultados
    speedup = t_seq / t_par if t_par > 0 else 0
    print("\n=== Resultados ===")
    print(f"Frames procesados: {total_frames}")
    print(f"Tiempo secuencial: {t_seq:.2f} s")
    print(f"Tiempo paralelo:   {t_par:.2f} s")
    print(f"Speedup:           {speedup:.2f}x\n")

    print("Videos generados correctamente.")
