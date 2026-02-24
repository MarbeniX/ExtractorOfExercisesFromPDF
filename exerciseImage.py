import cv2
import os
import numpy as np

def exerciseImage(imagePath, outputPath):
    if not os.path.exists(outputPath):
        print(f"Error: La imagen '{outputPath}' no existe.")
        return
    img = cv2.imread(imagePath)
    if img is None: return print(f"Error: No se encontró {imagePath}")

    height, width = img.shape[:2]
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

exerciseImage(imagePath="image2.png", outputPath="exercise_images")