import cv2
import numpy as np
import mediapipe as mp

counter = 0
stage = None


def calculate_angle(a, b, c):
    a = np.array(a)  # first
    b = np.array(b)  # mid
    c = np.array(c)  # center

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        result = pose.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img.flags.writeable = True

        # mp_pose.PoseLandmark.NOSE.value -> 0 - 32
        # x: 0.44315892457962036
        # y: 2.7824580669403076
        # z: -0.11315295845270157
        # visibility: 0.00031483417842537165

        try:
            landmarks = result.pose_landmarks.landmark

            # Get coordinates for angle_wrist
            left_index = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_pinky = [landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].y]

            # Get coordinates for angle_elbow
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

            # Calculate angle
            angle_wrist = calculate_angle(left_index, left_wrist, left_pinky)
            angle_elbow = calculate_angle(left_shoulder, left_elbow, left_wrist)

            # Wave patient attention
            if 30 < angle_elbow < 96:
                if counter == 4:
                    print("Patient needs attention")
                if counter == 4:
                    counter = 0
                if angle_wrist > 12:
                    stage = "down"
                if angle_wrist < 11 and stage == "down":
                    stage = "up"
                    counter += 1
                    print(counter)

            # Visualize to camera
            cv2.rectangle(img, (0, 0), (300, 110), (245, 117, 16), -1)

            cv2.putText(img, "ELBOW", (15, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, str(int(angle_elbow)), (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(img, "WRIST", (140, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, str(int(angle_wrist)), (143, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        except:
            pass

        mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('MediaPipe Feed', img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
