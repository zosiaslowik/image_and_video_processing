import os
import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

test_frames_dir = '/Users/natalia_pyzara/Desktop/train_test/test_data'
test_pose_data_dir = '/Users/natalia_pyzara/Desktop/test_pose_data_dir'
os.makedirs(test_pose_data_dir, exist_ok=True)

for pose_folder in os.listdir(test_frames_dir):
    pose_path = os.path.join(test_frames_dir, pose_folder)

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

                            np.save(os.path.join(test_pose_data_dir, f'{pose_folder}_{variation_folder}_{frame_file[:-4]}.npy'), landmarks)
