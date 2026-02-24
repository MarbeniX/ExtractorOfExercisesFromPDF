import cv2
import pytesseract
import re
import os
import numpy as np
import fitz  
from PIL import Image
import io

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
OUTPUT_FOLDER = "exercise_images"
if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)

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
                print(resultado)                
                # Imprimir resumen para validar
                
            except Exception as e:
                print(f"Error procesando segmento {i+1} de pág {num_pag+1}: {e}")
            finally:
                # Borrar el temporal para no llenar la carpeta de basura
                if os.path.exists(temp_name):
                    os.remove(temp_name)

def extract_exercise_info(image_path, outputPath):
    if not os.path.exists(outputPath):
        print(f"Error: La imagen '{outputPath}' no existe.")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Image not found"
    height, width = img.shape[:2]
    
    # Title
    title = extract_title(img, height, width)
    print(title)
    # Image
    extract_image(img, height, width, outputPath, image_path)
    # Muscle Groups
    muscle_groups = extract_muscle_groups(img, height, width)
    print(muscle_groups)
    # Description
    description = extract_description(img, height, width)
    print(description)
    # Difficulty
    difficulty = extract_difficulty(img, height, width)
    print(difficulty)


def extract_title(img, height, width):
    roi_title = img[0:int(height*0.12), 0:width]
    gray = cv2.cvtColor(roi_title, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    config_titulo = r'--oem 3 --psm 7'
    titulo = pytesseract.image_to_string(thresh, config=config_titulo)
    
    titulo_limpio = titulo.strip().replace('\n', ' ')
    if len(titulo_limpio) < 3:
        titulo_limpio = pytesseract.image_to_string(thresh, config='--psm 6').strip()

    return titulo_limpio

def extract_image(img, height, width, outputPath, imagePath): 
    right_roi = img[int(height*0.10):int(height*0.95), int(width*0.45):int(width*0.98)]

    gray = cv2.cvtColor(right_roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    border, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

    if border: 
        c = max(border, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        finalImage = right_roi[y:y+h, x:x+w]

        baseName = os.path.basename(imagePath)
        outputRouth = os.path.join(outputPath, baseName)
        cv2.imwrite(outputRouth, finalImage)
        print(f"Imagen recortada guardada en: {outputRouth}")
    else:
        print("No se detectó ningún contorno en la imagen.")

def extract_muscle_groups(img, height, width): 
    mapeo_groups = {
        "pectoral": "Pecho",
        "triceps": "Tríceps",
        "abdominales": "Abdomen",
        "hombros": "Hombros",
        "triceps": "Tríceps"
    }
    roi_izquierda = img[0:height, 0:int(width * 0.5)]
    roi_gray = cv2.cvtColor(roi_izquierda, cv2.COLOR_BGR2GRAY)
    texto = pytesseract.image_to_string(roi_gray).lower()
    musculos_encontrados = []

    texto_limpio = re.sub(r'[^\w\sáéíóúüñ]', '', texto)
    palabras_en_imagen = texto_limpio.split()

    for palabra in palabras_en_imagen:
        if palabra in mapeo_groups:
            nombre_oficial = mapeo_groups[palabra]
            if{"name": nombre_oficial} not in musculos_encontrados:
                musculos_encontrados.append(nombre_oficial)
    return musculos_encontrados

def extract_description(img, height, width): 
    roi_izquierda = img[0:height, 0:int(width * 0.5)]
    roi_gray = cv2.cvtColor(roi_izquierda, cv2.COLOR_BGR2GRAY)
    custom_config = r'--oem 3 --psm 6'
    texto_completo = pytesseract.image_to_string(roi_gray,  config=custom_config)
    patron = r"DESCRIPCI[OÓ]N[:\s]*(.*)"
    match = re.search(patron, texto_completo, re.IGNORECASE | re.DOTALL)

    if match:
        descripcion = match.group(1).strip()
        # Limpieza básica: quitar saltos de línea innecesarios si es un solo párrafo
        descripcion = descripcion.replace('\n', ' ').replace('  ', ' ')
        return descripcion
    else:
        return "No se encontró la sección DESCRIPCIÓN."

def extract_difficulty(img, height, width): 
    roi_izquierda = img[0:height, 0:int(width * 0.5)]
    roi_gray = cv2.cvtColor(roi_izquierda, cv2.COLOR_BGR2GRAY)

    data = pytesseract.image_to_data(roi_gray, output_type=pytesseract.Output.DICT)
    y_top, x_end, h_text = -1, -1, 0
    for i, word in enumerate(data['text']):
        if "DIFICULTAD" in word.upper():
            y_top = data['top'][i]
            x_end = data['left'][i] + data['width'][i] # Donde termina la palabra
            h_text = data['height'][i]
            break

    if y_top == -1:
        print("No se encontró la palabra DIFICULTAD")
        return 0
    roi_iconos = roi_gray[y_top-5 : y_top+h_text+10, x_end+5:]
    _, thresh = cv2.threshold(roi_iconos, 150, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((2,2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dificultad = 0
    for c in contornos:
        area = cv2.contourArea(c)
        # Filtramos por área: un icono de brazo suele tener más de 40-50 píxeles.
        # Esto ignora puntitos de polvo o restos de los dos puntos ":"
        if area > 30: 
            dificultad += 1
    return dificultad

# extract_exercise_info("image5.png", "exercise_images")
procesar_guia_completa("GuiaCalistenia.pdf", 21, 31)


