import face_recognition
import cv2
from matplotlib import pyplot as plt

# Cargar la imagen
img = cv2.imread("face.jpg")

# Convertir la imagen de BGR a RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Calcular las codificaciones de las caras en la imagen
encodings = face_recognition.face_encodings(img_rgb)

# Mostrar la gráfica de la primera codificación facial
if encodings:
    plt.figure(figsize=(8, 6))
    plt.plot(encodings[0])
    plt.title('Codificación facial')
    plt.xlabel('Características')
    plt.ylabel('Valor')
    plt.grid(True)
    plt.show()
else:
    print("No se encontraron caras en la imagen.")

# Detectar caras en la imagen
face_locations = face_recognition.face_locations(img_rgb)

# Mostrar la imagen con los vectores de codificación superpuestos
plt.imshow(img_rgb)
plt.title('Imagen con caras detectadas y codificaciones faciales')

# Dibujar rectángulos alrededor de las caras detectadas
for (top, right, bottom, left) in face_locations:
    plt.gca().add_patch(plt.Rectangle((left, top), right - left, bottom - top,
                                      linewidth=2, edgecolor='r', facecolor='none'))

# Función para guardar la imagen al presionar 's'
def press(event):
    if event.key == 's':
        plt.savefig('captura_imagen.png')
        print("Captura de la ventana guardada como 'captura_imagen.png'.")

# Conectar la función de presionar tecla al evento de la figura
plt.connect('key_press_event', press)

plt.show()
