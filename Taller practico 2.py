"""
Taller Práctico 2: Detección de Bordes con Algoritmo Sobel
Implementación Secuencial vs Paralelo (Multicore)
"""

import numpy as np
import cv2
import time
import multiprocessing as mp
from multiprocessing import Pool
import matplotlib.pyplot as plt

# ============================================================
# 1. ALGORITMO SECUENCIAL (1 CORE)
# ============================================================

def sobel_sequential(gray_image):
    """
    Implementación secuencial del algoritmo Sobel
    """
    height, width = gray_image.shape
    output = np.zeros_like(gray_image, dtype=np.float32)
    
    # Kernels de Sobel
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    
    Ky = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]], dtype=np.float32)
    
    # Procesar cada píxel (excluyendo bordes)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Extraer ventana 3x3
            window = gray_image[i-1:i+2, j-1:j+2]
            
            # Calcular Gx (derivada horizontal)
            Gx = 0
            for ki in range(3):
                for kj in range(3):
                    Gx += window[ki, kj] * Kx[ki, kj]
            
            # Calcular Gy (derivada vertical)
            Gy = 0
            for ki in range(3):
                for kj in range(3):
                    Gy += window[ki, kj] * Ky[ki, kj]
            
            # Magnitud del gradiente
            magnitude = np.sqrt(Gx**2 + Gy**2)
            output[i, j] = min(255, magnitude)
    
    return output.astype(np.uint8)


# ============================================================
# 2. ALGORITMO PARALELO (MULTICORE)
# ============================================================

def process_chunk(args):
    """
    Procesa un chunk (porción) de la imagen
    """
    gray_image, start_row, end_row = args
    height, width = gray_image.shape
    chunk_output = np.zeros((end_row - start_row, width), dtype=np.float32)
    
    # Kernels de Sobel
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    
    Ky = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]], dtype=np.float32)
    
    # Procesar solo las filas asignadas a este chunk
    for i in range(start_row, end_row):
        for j in range(1, width - 1):
            # Extraer ventana 3x3
            window = gray_image[i-1:i+2, j-1:j+2]
            
            # Calcular Gx
            Gx = 0
            for ki in range(3):
                for kj in range(3):
                    Gx += window[ki, kj] * Kx[ki, kj]
            
            # Calcular Gy
            Gy = 0
            for ki in range(3):
                for kj in range(3):
                    Gy += window[ki, kj] * Ky[ki, kj]
            
            # Magnitud del gradiente
            magnitude = np.sqrt(Gx**2 + Gy**2)
            chunk_output[i - start_row, j] = min(255, magnitude)
    
    return chunk_output, start_row, end_row


def sobel_parallel(gray_image, num_cores=None):
    """
    Implementación paralela del algoritmo Sobel usando multiprocessing
    """
    if num_cores is None:
        num_cores = mp.cpu_count()
    
    height, width = gray_image.shape
    output = np.zeros_like(gray_image, dtype=np.float32)
    
    # Dividir el trabajo en chunks (excluyendo primera y última fila)
    rows_to_process = height - 2  # Excluir bordes
    rows_per_core = rows_to_process // num_cores
    
    # Crear argumentos para cada proceso
    chunks = []
    for i in range(num_cores):
        start_row = 1 + i * rows_per_core
        end_row = start_row + rows_per_core if i < num_cores - 1 else height - 1
        chunks.append((gray_image, start_row, end_row))
    
    # Procesar en paralelo
    with Pool(processes=num_cores) as pool:
        results = pool.map(process_chunk, chunks)
    
    # Combinar resultados
    for chunk_output, start_row, end_row in results:
        output[start_row:end_row, :] = chunk_output
    
    return output.astype(np.uint8)


# ============================================================
# 3. FUNCIÓN PRINCIPAL Y PRUEBAS
# ============================================================

def main():
    """
    Función principal para probar los algoritmos
    """
    # Cargar imagen
    print("Cargando imagen...")
    image_path = "imagen_prueba.jpg"  # Cambia por tu imagen
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: No se pudo cargar la imagen.")
        print("Asegúrate de tener una imagen llamada 'imagen_prueba.jpg'")
        return
    
    # Convertir a escala de grises
    print("Convirtiendo a escala de grises...")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Obtener dimensiones
    height, width = gray_image.shape
    print(f"Dimensiones de la imagen: {width}x{height}")
    
    # ============================================================
    # ALGORITMO SECUENCIAL
    # ============================================================
    print("\n" + "="*60)
    print("EJECUTANDO ALGORITMO SECUENCIAL (1 CORE)")
    print("="*60)
    
    start_time = time.time()
    sequential_result = sobel_sequential(gray_image)
    sequential_time = time.time() - start_time
    
    print(f"Tiempo de ejecución: {sequential_time:.4f} segundos")
    
    # ============================================================
    # ALGORITMO PARALELO
    # ============================================================
    print("\n" + "="*60)
    print("EJECUTANDO ALGORITMO PARALELO (MULTICORE)")
    print("="*60)
    
    num_cores = mp.cpu_count()
    print(f"Número de cores disponibles: {num_cores}")
    
    start_time = time.time()
    parallel_result = sobel_parallel(gray_image, num_cores)
    parallel_time = time.time() - start_time
    
    print(f"Tiempo de ejecución: {parallel_time:.4f} segundos")
    
    # ============================================================
    # ANÁLISIS DE RENDIMIENTO
    # ============================================================
    print("\n" + "="*60)
    print("ANÁLISIS DE RENDIMIENTO")
    print("="*60)
    
    speedup = sequential_time / parallel_time
    efficiency = (speedup / num_cores) * 100
    
    print(f"Tiempo Secuencial:  {sequential_time:.4f} segundos")
    print(f"Tiempo Paralelo:    {parallel_time:.4f} segundos")
    print(f"Speedup:            {speedup:.2f}x")
    print(f"Eficiencia:         {efficiency:.2f}%")
    print(f"Cores utilizados:   {num_cores}")
    
    # ============================================================
    # GUARDAR RESULTADOS
    # ============================================================
    print("\nGuardando resultados...")
    cv2.imwrite("resultado_original.jpg", image)
    cv2.imwrite("resultado_gris.jpg", gray_image)
    cv2.imwrite("resultado_secuencial.jpg", sequential_result)
    cv2.imwrite("resultado_paralelo.jpg", parallel_result)
    
    # ============================================================
    # VISUALIZACIÓN
    # ============================================================
    print("Generando visualización...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Imagen Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gray_image, cmap='gray')
    axes[0, 1].set_title('Escala de Grises')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(sequential_result, cmap='gray')
    axes[1, 0].set_title(f'Sobel Secuencial\n({sequential_time:.4f}s)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(parallel_result, cmap='gray')
    axes[1, 1].set_title(f'Sobel Paralelo\n({parallel_time:.4f}s - Speedup: {speedup:.2f}x)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('comparacion_resultados.png', dpi=300, bbox_inches='tight')
    print("Visualización guardada como 'comparacion_resultados.png'")
    
    plt.show()
    
    print("\n✅ Proceso completado exitosamente!")


if __name__ == "__main__":
    main()