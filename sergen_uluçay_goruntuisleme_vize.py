import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Parmak pozisyonu kontrol fonksiyonu
def koordinat_getir(landmarks, indeks, h, w):
    landmark = landmarks[indeks]
    return int(landmark.x * w), int(landmark.y * h)

def fingers_up(landmarks, handedness, h, w):
    fingers = []

    # İşaret
    y_tip, y_pip = koordinat_getir(landmarks, 8, h, w)[1], koordinat_getir(landmarks, 6, h, w)[1]
    fingers.append(1 if y_tip < y_pip else 0)

    # Orta
    y_tip, y_pip = koordinat_getir(landmarks, 12, h, w)[1], koordinat_getir(landmarks, 10, h, w)[1]
    fingers.append(1 if y_tip < y_pip else 0)

    # Yüzük
    y_tip, y_pip = koordinat_getir(landmarks, 16, h, w)[1], koordinat_getir(landmarks, 14, h, w)[1]
    fingers.append(1 if y_tip < y_pip else 0)

    # Serçe
    y_tip, y_pip = koordinat_getir(landmarks, 20, h, w)[1], koordinat_getir(landmarks, 18, h, w)[1]
    fingers.append(1 if y_tip < y_pip else 0)

    # Baş parmak
    x_tip, x_ip = koordinat_getir(landmarks, 4, h, w)[0], koordinat_getir(landmarks, 2, h, w)[0]
    if handedness == "Right":
        fingers.append(1 if x_tip > x_ip else 0)
    else:
        fingers.append(1 if x_tip < x_ip else 0)

    return sum(fingers)

# Model ve detector tanımı
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# Kamera başlat
cam = cv2.VideoCapture(0)

while cam.isOpened():
    basari, frame = cam.read()
    if not basari:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    detection_result = detector.detect(mp_image)

    h, w, _ = frame.shape
    annotated_image = np.copy(frame_rgb)

    for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
        handedness = detection_result.handedness[idx][0].category_name
        count = fingers_up(hand_landmarks, handedness, h, w)

        # Landmarkları çiz
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style()
        )

        # Ekrana el yönü ve parmak sayısını yaz
        cv2.putText(annotated_image, f"{handedness}: {count}", (10, 60 + idx * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Parmak Sayaci", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
