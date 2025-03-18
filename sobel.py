import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

# Cargar la imagen en escala de grises
imagen = cv2.imread('placa.png', cv2.IMREAD_GRAYSCALE)

# Aplicar el operador de Sobel en X y Y
sobel_x = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)

imagen_invertida = cv2.bitwise_not(imagen)
_, umbralizada = cv2.threshold(imagen_invertida, 215, 255, cv2.THRESH_BINARY)
kernel = np.ones((3, 3), np.uint8)
apertura = cv2.morphologyEx(umbralizada, cv2.MORPH_OPEN, kernel)
cierre = cv2.morphologyEx(apertura, cv2.MORPH_CLOSE, kernel)


# Convertir los resultados a valores absolutos para visualización
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)
cierre = cv2.convertScaleAbs(cierre)

texto = pytesseract.image_to_string(cierre)
print("Texto detectado:", texto)

# Mostrar los resultados
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 2)
plt.imshow(sobel_x, cmap='gray')
plt.title('Sobel en X')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sobel_y, cmap='gray')
plt.title('Sobel en Y')
plt.axis('off')

plt.subplot(1, 3, 1)
plt.imshow(apertura, cmap='gray')
plt.title('Prueba')
plt.axis('off')

plt.tight_layout()
plt.show()




