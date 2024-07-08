import cv2
import numpy as np
import dlib
import time

# Define los puntos de las marcas faciales
JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_BRIDGE_POINTS = list(range(27, 31))
LOWER_NOSE_POINTS = list(range(31, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))

def draw_shape_lines_all(np_shape, image):
    draw_shape_lines_range(np_shape, image, JAWLINE_POINTS)
    draw_shape_lines_range(np_shape, image, RIGHT_EYEBROW_POINTS)
    draw_shape_lines_range(np_shape, image, LEFT_EYEBROW_POINTS)
    draw_shape_lines_range(np_shape, image, NOSE_BRIDGE_POINTS)
    draw_shape_lines_range(np_shape, image, LOWER_NOSE_POINTS)
    draw_shape_lines_range(np_shape, image, RIGHT_EYE_POINTS, True)
    draw_shape_lines_range(np_shape, image, LEFT_EYE_POINTS, True)
    draw_shape_lines_range(np_shape, image, MOUTH_OUTLINE_POINTS, True)
    draw_shape_lines_range(np_shape, image, MOUTH_INNER_POINTS, True)

def draw_shape_lines_range(np_shape, image, range_points, is_closed=False):
    np_shape_display = np_shape[range_points]
    points = np.array(np_shape_display, dtype=np.int32)
    cv2.polylines(image, [points], is_closed, (255, 255, 0), thickness=1, lineType=cv2.LINE_8)

def draw_shape_points(np_shape, image):
    for (x, y) in np_shape:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

def shape_to_np(dlib_shape, dtype="int"):
    coordinates = np.zeros((dlib_shape.num_parts, 2), dtype=dtype)
    for i in range(0, dlib_shape.num_parts):
        coordinates[i] = (dlib_shape.part(i).x, dlib_shape.part(i).y)
    return coordinates

# Inicializar detector de caras y predictor de marcas faciales de dlib
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

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
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detección de rostros
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=(30, 30))

    for (x, y, w, h) in faces:
        x, y, w, h = [v * 2 for v in (x, y, w, h)]  # Volver a escalar las coordenadas
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Convertir a escala de grises para dlib
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), dlib_rect)
        shape = shape_to_np(shape)
        draw_shape_lines_all(shape, frame)

    cv2.putText(frame, f"Rostros detectados: {len(faces)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Detector de Rostros', frame)

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

    if cv2.getWindowProperty('Detector de Rostros', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
