import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


def get_finger_direction(hand_landmarks):
    base = hand_landmarks.landmark[5]
    tip = hand_landmarks.landmark[8]

    dx = tip.x - base.x
    dy = tip.y - base.y

    threshold = 0.08

    if abs(dx) < threshold and abs(dy) < threshold:
        return "DUR"

    if abs(dx) > abs(dy):
        if dx > 0:
            return "SAG"
        else:
            return "SOL"

    else:
        if dy < 0:
            return "ILERI"
        else:
            return "GERI"


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera acilamadi.")
    exit()


while True:
    success, frame = cap.read()

    if not success:
        print("Kameradan goruntu alinamadi.")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)

    command = "DUR"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            command = get_finger_direction(hand_landmarks)

    cv2.putText(
        frame,
        f"Command: {command}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Sumo Robot Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
