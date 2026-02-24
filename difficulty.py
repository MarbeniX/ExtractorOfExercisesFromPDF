import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def contar_dificultad_por_manchas(ruta_imagen):
    # 1. Cargar imagen
    img_rgb = cv2.imread(ruta_imagen)
    if img_rgb is None: return 0
    
    alto, ancho = img_rgb.shape[:2]
    
    # 2. Enfocarnos solo en la mitad izquierda (donde está el texto)
    roi_izquierda = img_rgb[0:alto, 0:int(ancho * 0.5)]
    gray = cv2.cvtColor(roi_izquierda, cv2.COLOR_BGR2GRAY)

    # 3. Localizar la palabra "DIFICULTAD" para saber en qué altura buscar
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    
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

    # 4. Crear una franja de búsqueda a la derecha de la palabra detectada
    # Tomamos desde donde termina la palabra hasta el final de la ROI
    # y le damos un margen vertical para capturar los iconos
    roi_iconos = gray[y_top-5 : y_top+h_text+10, x_end+5:]

    # 5. Segmentar las manchas negras
    # Convertimos a blanco y negro puro e invertimos: 
    # El fondo será negro y los iconos (las manchas) serán blancos.
    _, thresh = cv2.threshold(roi_iconos, 150, 255, cv2.THRESH_BINARY_INV)

    # 6. Limpieza: Eliminar ruidos pequeños (como los puntos de los ":")
    kernel = np.ones((2,2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 7. Contar las manchas (Contornos)
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dificultad = 0
    for c in contornos:
        area = cv2.contourArea(c)
        # Filtramos por área: un icono de brazo suele tener más de 40-50 píxeles.
        # Esto ignora puntitos de polvo o restos de los dos puntos ":"
        if area > 30: 
            dificultad += 1

    print(f"Dificultad detectada (por manchas): {dificultad}")
    return dificultad

# Prueba
contar_dificultad_por_manchas("img7.png")