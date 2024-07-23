# aI_Based_Video_SelfModeling_Program
#1정 연수 코드 공유


import cv2
import numpy as np
import time
import mediapipe as mp

# MediaPipe 손 모델 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

# 전역 변수 초기화
mode = -1
drawing_color = (0, 0, 255)  # 기본 색상은 빨간색
canvas = None
time_canvas = None
buttons = [
    {"name": "Clear", "color": (0, 0, 0)},
    {"name": "Red", "color": (0, 0, 255)},
    {"name": "Blue", "color": (255, 0, 0)}
]
button_height = 50
object_detection_enabled = True
object_detection_threshold = 0.5
fist_start_time = None  # 주먹 쥐기 시작한 시간을 저장

# 얼굴 검출 모델 로드 함수
def load_face_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 얼굴 검출 함수
def detect_faces(frame, face_detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# 극단적 모자이크 처리 함수
def extreme_mosaic(frame, face, min_mosaic_size, max_mosaic_size, color_variation):
    (x, y, w, h) = face
    face_roi = frame[y:y+h, x:x+w]
    mosaic_size = np.random.randint(min_mosaic_size, max_mosaic_size)
    face_roi = cv2.resize(face_roi, (mosaic_size, mosaic_size), interpolation=cv2.INTER_LINEAR)
    face_roi = cv2.resize(face_roi, (w, h), interpolation=cv2.INTER_NEAREST)
    face_roi = face_roi + np.random.randint(-color_variation, color_variation, face_roi.shape, dtype=np.int16)
    face_roi = np.clip(face_roi, 0, 255).astype(np.uint8)
    frame[y:y+h, x:x+w] = face_roi
    return frame

# 객체 검출 모델 로드 함수
def load_object_detector():
    # 여기에서 파일 경로를 올바르게 설정합니다.
    prototxt_path = "/Users/ibyeonghwi/Desktop/파이썬/파이썬 코드/1정연수/ssd_mobilenet_v2_coco.prototxt"
    caffemodel_path = "/Users/ibyeonghwi/Desktop/파이썬/파이썬 코드/1정연수/ssd_mobilenet_v2_coco.caffemodel"
    net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
    return net

# 객체 검출 함수
def detect_objects(frame, net, threshold):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    objects = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > threshold:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            objects.append((idx, confidence, (startX, startY, endX - startX, endY - startY)))
    return objects

# 객체 레이블 딕셔너리 (COCO 데이터셋)
object_labels = {
    0: "background",
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush"
}
# 선 그리기 함수
def draw_line_with_time(canvas, time_canvas, start, end, color, thickness, current_time):
    cv2.line(canvas, start, end, color, thickness)
    cv2.line(time_canvas, start, end, current_time, thickness)

# 느낌표 모양 그리기 함수
def draw_exclamation_mark(canvas, size):
    center = (size - 50, 50)
    thickness = 10
    # 느낌표 그리기
    cv2.circle(canvas, center, 15, (0, 0, 255), thickness)
    cv2.line(canvas, (center[0], center[1] + 20), (center[0], center[1] + 40), (0, 0, 255), thickness)



# 버튼 생성 함수
def create_buttons(frame, mode, active_button=None):
    button_width = frame.shape[1] // len(buttons)
    for i, button in enumerate(buttons):
        x_start = i * button_width
        color = (0, 255, 0) if i == active_button else button["color"]
        cv2.rectangle(frame, (x_start, 0), (x_start + button_width, button_height), color, -1)
        cv2.putText(frame, button["name"], (x_start + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 그리기 모드일 때 연필 모양 표시
    if mode == 0:
        pencil_img = cv2.imread("pencil.png", cv2.IMREAD_UNCHANGED)
        if pencil_img is not None:
            pencil_img = cv2.resize(pencil_img, (button_height, button_height))
            frame[0:button_height, 0:button_height] = blend_transparent(frame[0:button_height, 0:button_height], pencil_img)
    
    return frame

    



# 이미지를 투명하게 합성하기 위한 함수
def blend_transparent(frame, overlay):
    overlay_img = overlay[:,:,:3]  # RGB 채널만 가져오기

    if overlay.shape[2] == 4:  # Alpha 채널이 있는 경우
        overlay_mask = overlay[:,:,3]  # Alpha 채널 가져오기
    else:
        overlay_mask = np.ones_like(overlay_img[:,:,0]) * 255  # Alpha 채널 생성 (모두 불투명)

    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = 255 - overlay_mask

    frame_part = (frame * (background_mask / 255.0)).astype(np.uint8)
    overlay_part = (overlay_img * (overlay_mask / 255.0)).astype(np.uint8)

    return cv2.add(frame_part, overlay_part)



# 버튼 클릭 처리 함수
def handle_button(button_index):
    global mode, drawing_color, canvas

    if button_index == 0:  # Clear
        canvas = np.zeros_like(canvas)
    elif button_index == 1:  # Red
        drawing_color = (0, 0, 255)
    elif button_index == 2:  # Blue
        drawing_color = (255, 0, 0)
    
    mode = -1  # 선택 후 모드를 초기화
    
# 손가락 하나가 펴짐 확인 함수
def is_one_finger_open(hand_landmarks):
    fingers_open = [
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y
    ]
    return fingers_open.count(True) == 1

# 주먹 쥐기 확인 함수
def is_fist_closed(hand_landmarks):
    fingers_open = [
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y,
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y
    ]
    return fingers_open.count(False) == 5

# 메인 함수
def main():
    global canvas, time_canvas, mode  # Ensure mode is global

    # 웹캠 설정 (Camo 앱을 통해 연결된 아이폰 카메라 사용)
    cap = cv2.VideoCapture(1)  # 필요에 따라 인덱스 조정
    face_detector = load_face_detector()
    object_detector = load_object_detector()
    
    # 초기 캔버스 생성
    ret, frame = cap.read()
    if not ret:
        print("카메라를 열 수 없습니다.")
        return

    canvas = np.zeros_like(frame)
    time_canvas = np.zeros_like(frame)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 얼굴 검출
        faces = detect_faces(frame, face_detector)
        for face in faces:
            frame = extreme_mosaic(frame, face, min_mosaic_size=10, max_mosaic_size=30, color_variation=50)
        
        # 객체 검출
        objects = detect_objects(frame, object_detector, object_detection_threshold)
        for obj in objects:
            idx, confidence, (x, y, w, h) = obj
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        # 손 검출 및 버튼 인터페이스 처리
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if is_one_finger_open(hand_landmarks):
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    x = int(index_finger_tip.x * frame.shape[1])
                    y = int(index_finger_tip.y * frame.shape[0])
                    if y < button_height:
                        button_index = x // (frame.shape[1] // len(buttons))
                        handle_button(button_index)
                    elif mode == 0:
                        cv2.circle(frame, (x, y), 5, drawing_color, -1)
                        cv2.circle(canvas, (x, y), 5, drawing_color, -1)
                        cv2.circle(time_canvas, (x, y), 5, (0, 255, 0), -1)  # Time canvas에 녹색으로 그림

        # 버튼 생성 및 캔버스 결합
        frame = create_buttons(frame, mode)
        combined_frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

        cv2.imshow("Frame", combined_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC 키를 누르면 종료
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
