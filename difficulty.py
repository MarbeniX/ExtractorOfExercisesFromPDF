import cv2
import numpy as np

def count_muscle_emojis(image_path, template_path='muscleTemplate.png'):
    # 1. Cargar imágenes
    img_rgb = cv2.imread('image.png')
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    # Asegúrate de tener este recorte pequeño del brazo
    template = cv2.imread('muscleTemplate.png', 0) 

    w, h = template.shape[::-1]

    # 2. Ejecutar la búsqueda
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

    # 3. Filtrar por coincidencia (80%)
    threshold = 0.8
    loc = np.where(res >= threshold)

    # 4. Convertir puntos a rectángulos para agrupar (evita contar el mismo emoji 100 veces)
    rects = []
    for pt in zip(*loc[::-1]):
        rects.append([int(pt[0]), int(pt[1]), int(w), int(h)])

    rects_agrupados, _ = cv2.groupRectangles(rects, groupThreshold=1, eps=0.5)
    print(len(rects_agrupados))

count_muscle_emojis(image_path="image.png")