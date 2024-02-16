import os
import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

train_frames_dir = '/Users/natalia_pyzara/Desktop/train_test/train_data'
pose_data_dir = '/Users/natalia_pyzara/Desktop/train_pose_data_dir'
os.makedirs(pose_data_dir, exist_ok=True)

for pose_folder in os.listdir(train_frames_dir):
    pose_path = os.path.join(train_frames_dir, pose_folder)

    if os.path.isdir(pose_path):
        for variation_folder in os.listdir(pose_path):
            variation_path = os.path.join(pose_path, variation_folder)

            if os.path.isdir(variation_path):
                for frame_file in os.listdir(variation_path):
                    frame_path = os.path.join(variation_path, frame_file)

                    if os.path.isfile(frame_path):
                        frame = cv2.imread(frame_path)

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        results = pose.process(frame_rgb)

                        if results.pose_landmarks:
                            landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.pose_landmarks.landmark]).flatten()

                            np.save(os.path.join(pose_data_dir, f'{pose_folder}_{variation_folder}_{frame_file[:-4]}.npy'), landmarks)