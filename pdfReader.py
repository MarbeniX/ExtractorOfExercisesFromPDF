import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import cv2
import io
import os

def procesar_guia_completa(ruta_pdf, p_inicio, p_fin, carpeta_imagenes="exercise_images"):
    # 1. Asegurar que la carpeta de destino exista
    if not os.path.exists(carpeta_imagenes):
        os.makedirs(carpeta_imagenes)

    doc = fitz.open(ruta_pdf)
    
    # Ajuste de índices (PDF empieza en 0)
    p_inicio = max(0, p_inicio - 1)
    p_fin = min(len(doc), p_fin)

    for num_pag in range(p_inicio, p_fin):
        pagina = doc.load_page(num_pag)
        
        # Aumentamos la resolución a 300 DPI (zoom 4) para que el OCR y 
        # el conteo de manchas de dificultad sean mucho más precisos.
        matriz = fitz.Matrix(4.0, 4.0) 
        pix = pagina.get_pixmap(matrix=matriz)
        
        # Convertir bytes del PDF a imagen de OpenCV (BGR)
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        img_pag_completa = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
        
        alto, ancho = img_pag_completa.shape[:2]
        alto_segmento = alto // 3
        
        print(f"\n--- Leyendo Página {num_pag + 1} ---")
        
        for i in range(3):
            # 2. Segmentación geométrica de los 3 ejercicios
            y0 = i * alto_segmento
            y1 = (i + 1) * alto_segmento if i < 2 else alto
            
            # Recorte del ejercicio actual
            ejercicio_roi = img_pag_completa[y0:y1, :]
            
            # 3. Guardar temporalmente para la función (si es que requiere ruta de archivo)
            temp_name = f"temp_p{num_pag+1}_seg{i+1}.png"
            cv2.imwrite(temp_name, ejercicio_roi)
            
            # 4. LLAMADA A TU FUNCIÓN EXTRACTORA
            try:
                print(f"Procesando ejercicio {i+1}...")
                # Pasamos la ruta temporal y la carpeta donde guardará el recorte del dibujo
                resultado = extract_exercise_info(temp_name, carpeta_imagenes)
                
                # Imprimir resumen para validar
                print(f"Exito: {resultado.get('titulo', 'Sin Titulo')}")
                
            except Exception as e:
                print(f"Error procesando segmento {i+1} de pág {num_pag+1}: {e}")
            finally:
                # Borrar el temporal para no llenar la carpeta de basura
                if os.path.exists(temp_name):
                    os.remove(temp_name)

# --- Para ejecutarlo ---
# procesar_guia_completa("tu_archivo.pdf", 26, 30)