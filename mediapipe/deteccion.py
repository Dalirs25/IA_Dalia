import cv2 
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Puntas de los índices
    left_index = None
    right_index = None

    # Cuadro azul por defecto (200x200), posición fija
    blue_tl = (100, 100)
    lado_azul = 200  # se actualizará si hay dos manos

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # 'Left' o 'Right'
            h, w, _ = frame.shape

            # Índice (landmark 8)
            idx = hand_landmarks.landmark[8]
            x, y = int(idx.x * w), int(idx.y * h)

            if label == 'Left':
                left_index = (x, y)
            elif label == 'Right':
                right_index = (x, y)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if left_index and right_index:
            # Rectángulo verde controlado por dedos
            x_min = min(left_index[0], right_index[0])
            y_min = min(left_index[1], right_index[1])
            x_max = max(left_index[0], right_index[0])
            y_max = max(left_index[1], right_index[1])

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
            cv2.circle(frame, left_index, 8, (255, 0, 0), -1)
            cv2.circle(frame, right_index, 8, (0, 0, 255), -1)

            # === NUEVO: azul siempre cuadrado y del mismo "escala" del verde ===
            w_box = max(1, x_max - x_min)
            h_box = max(1, y_max - y_min)
            lado_azul = max(w_box, h_box)  # mantiene cuadrado

    # Dibuja el cuadro azul (cuadrado), solo escala el tamaño, posición fija
    blue_br = (blue_tl[0] + int(lado_azul), blue_tl[1] + int(lado_azul))
    cv2.rectangle(frame, blue_tl, blue_br, (255, 0, 0), 3)

    cv2.imshow("Line", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
