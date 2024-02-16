import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

frame = cv2.imread('/Users/natalia_pyzara/Desktop/output_folder/output_folder_wheel_pose/wheel3/frame_0001.jpg')
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

results = pose.process(frame_rgb)

black_background = np.zeros_like(frame)

if results.pose_landmarks:
    for landmark_id, landmark in enumerate(results.pose_landmarks.landmark):
        landmark_x = int(landmark.x * frame.shape[1])
        landmark_y = int(landmark.y * frame.shape[0])
        landmark_z = landmark.z if landmark.HasField('z') else None

        cv2.circle(black_background, (landmark_x, landmark_y), 5, (0, 255, 0), -1)  # Green circle


mp_drawing = mp.solutions.drawing_utils
for connection in mp_pose.POSE_CONNECTIONS:
    start_point = tuple(np.multiply((results.pose_landmarks.landmark[connection[0]].x, results.pose_landmarks.landmark[connection[0]].y), [frame.shape[1], frame.shape[0]]).astype(int))
    end_point = tuple(np.multiply((results.pose_landmarks.landmark[connection[1]].x, results.pose_landmarks.landmark[connection[1]].y), [frame.shape[1], frame.shape[0]]).astype(int))
    cv2.line(black_background, start_point, end_point, (0, 255, 0), 2)


cv2.imwrite('/Users/natalia_pyzara/Desktop/green_skeleton_output.jpg', black_background)

cv2.imshow('Yoga Pose Detection', black_background)
cv2.waitKey(0)
cv2.destroyAllWindows()