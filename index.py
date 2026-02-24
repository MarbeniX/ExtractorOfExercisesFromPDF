import cv2
import pytesseract
import re

# SOLO si estás en Windows, indica la ruta
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Leer imagen
imagen = cv2.imread("image.png")

# Convertir a gris (mejora OCR)
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Extraer texto
texto = pytesseract.image_to_string(gris)

print("Texto detectado:")
print(texto)

# Buscar la línea que contiene DIFICULTAD
if "DIFICULTAD" in texto:
    linea = [l for l in texto.split("\n") if "DIFICULTAD" in l][0]
    
    # Contar emojis (unicode rango general)
    emojis = re.findall(r'[^\w\s:]', linea)
    print("Número de emojis:", len(emojis))
else:
    print("No se encontró la palabra DIFICULTAD")