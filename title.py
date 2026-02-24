import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_title(image_path): 
    img = cv2.imread(image_path)
    if img is None: return "Error: Image not found"

    alto, ancho = img.shape[:2]
    # 1. RECORTAR LA FRANJA SUPERIOR
    # Tomamos el primer 15% de la altura de la imagen, donde suele estar el título
    y_limite = int(alto * 0.15)
    franja_superior = img[0:y_limite, 0:ancho]

    gray = cv2.cvtColor(franja_superior, cv2.COLOR_BGR2GRAY)
    # Aumentar contraste (esto ayuda mucho con fuentes gruesas como las de tus imágenes)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. OCR ESPECÍFICO PARA TÍTULOS
    # PSM 7: Trata la imagen como una sola línea de texto
    # PSM 8: Trata la imagen como una sola palabra (útil si el título es corto)
    config_titulo = r'--oem 3 --psm 7'
    titulo = pytesseract.image_to_string(thresh, config=config_titulo)

    titulo_limpio = titulo.strip().replace('\n', ' ')
    if len(titulo_limpio) < 3:
        titulo_limpio = pytesseract.image_to_string(thresh, lang='spa', config='--psm 6').strip()

    return titulo_limpio

resultado = extract_title("image2.png")
print(resultado)