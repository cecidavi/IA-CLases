import cv2
import time

# Cargar el clasificador de Haar para rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Iniciar captura de video
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

# Parámetros ajustables
scaleFactor = 1.1
minNeighbors = 5

# Función para guardar la captura de pantalla
def guardar_captura(frame):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"captura_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Captura guardada como {filename}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar la imagen.")
        break

    # Reducir la resolución del frame para mejorar el rendimiento
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    # Aplicar suavizado a la imagen en escala de grises
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detección de rostros
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=(30, 30))

    # Dibujar rectángulos alrededor de los rostros detectados
    for (x, y, w, h) in faces:
        x, y, w, h = [v * 2 for v in (x, y, w, h)]  # Volver a escalar las coordenadas
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Mostrar el número de rostros detectados
    cv2.putText(frame, f"Rostros detectados: {len(faces)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Mostrar la imagen con las detecciones
    cv2.imshow('Detector de Rostros', frame)

    # Captura de pantalla y ajustes de parámetros con teclas
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        guardar_captura(frame)
    elif key == ord('+'):
        scaleFactor += 0.1
    elif key == ord('-'):
        scaleFactor = max(1.1, scaleFactor - 0.1)
    elif key == ord('n'):
        minNeighbors += 1
    elif key == ord('m'):
        minNeighbors = max(1, minNeighbors - 1)

    # Comprobar si la ventana fue cerrada
    if cv2.getWindowProperty('Detector de Rostros', cv2.WND_PROP_VISIBLE) < 1:
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()