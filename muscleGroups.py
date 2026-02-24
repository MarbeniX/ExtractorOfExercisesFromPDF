# export const muscleGroups = [
#     { name: "Pecho" },
#     { name: "Espalda" },
#     { name: "Piernas" },
#     { name: "Hombros" },
#     { name: "Bíceps" },
#     { name: "Tríceps" },
#     { name: "Abdomen" },
#     { name: "Glúteos" },
#     { name: "Antebrazos" },
# ];


import cv2
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# muscles = { "Pectoral", "tríceps", "abdominales", "Tríceps", "hombros"}

def interpretar_musculos(image_path): 
    mapeo_groups = {
        "pectoral": "Pecho",
        "triceps": "Tríceps",
        "abdominales": "Abdomen",
        "hombros": "Hombros",
        "triceps": "Tríceps"
    }

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    texto = pytesseract.image_to_string(gray).lower()
    musculos_encontrados = []

    texto_limpio = re.sub(r'[^\w\sáéíóúüñ]', '', texto)
    palabras_en_imagen = texto_limpio.split()

    for palabra in palabras_en_imagen:
        if palabra in mapeo_groups:
            nombre_oficial = mapeo_groups[palabra]
            if{"name": nombre_oficial} not in musculos_encontrados:
                musculos_encontrados.append(nombre_oficial)
    return musculos_encontrados

resultado = interpretar_musculos("image9.png")
print(resultado)