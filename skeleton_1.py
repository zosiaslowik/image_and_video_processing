import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

frame = cv2.imread('/Users/natalia_pyzara/Desktop/output_folder/output_folder_wheel_pose/wheel3/frame_0001.jpg')
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

results = pose.process(frame_rgb)

if results.pose_landmarks:
    for landmark_id, landmark in enumerate(results.pose_landmarks.landmark):
        landmark_x = landmark.x
        landmark_y = landmark.y
        landmark_z = landmark.z if landmark.HasField('z') else None

        print(f"Landmark {landmark_id}: X={landmark_x}, Y={landmark_y}, Z={landmark_z}")

mp_drawing = mp.solutions.drawing_utils

# Change the color of the skeleton (e.g., to red)
draw_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, draw_spec, draw_spec)

cv2.imshow('Yoga Pose Detection', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
