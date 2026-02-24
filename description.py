import cv2
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_desciption(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Image not found"

    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Usamos el modo de segmentación de página 6 (PSM 6) 
    # que asume un bloque de texto uniforme.
    custom_config = r'--oem 3 --psm 6'
    texto_completo = pytesseract.image_to_string(gray,  config=custom_config)

    patron = r"DESCRIPCI[OÓ]N[:\s]*(.*)"
    match = re.search(patron, texto_completo, re.IGNORECASE | re.DOTALL)

    if match:
        descripcion = match.group(1).strip()
        # Limpieza básica: quitar saltos de línea innecesarios si es un solo párrafo
        descripcion = descripcion.replace('\n', ' ').replace('  ', ' ')
        return descripcion
    else:
        return "No se encontró la sección DESCRIPCIÓN."

texto_final = extract_desciption("image2.png")
print(texto_final)
