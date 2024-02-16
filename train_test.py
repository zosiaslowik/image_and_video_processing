import os
from sklearn.model_selection import train_test_split
import shutil

output_root_folder = "/Users/natalia_pyzara/Desktop/output_folder"

train_test_folder = "/Users/natalia_pyzara/Desktop/train_test"

if os.path.exists(train_test_folder):
    shutil.rmtree(train_test_folder)

os.makedirs(train_test_folder)

for subfolder in os.listdir(output_root_folder):
    subfolder_path = os.path.join(output_root_folder, subfolder)
    if os.path.isdir(subfolder_path):
        destination_path = os.path.join(train_test_folder, subfolder)
        shutil.move(subfolder_path, destination_path)

train_folder = "/Users/natalia_pyzara/Desktop/train_data"
test_folder = "/Users/natalia_pyzara/Desktop/test_data"

if os.path.exists(train_folder):
    shutil.rmtree(train_folder)
if os.path.exists(test_folder):
    shutil.rmtree(test_folder)

os.makedirs(train_folder)
os.makedirs(test_folder)

all_yoga_poses = os.listdir(train_test_folder)

for yoga_pose in all_yoga_poses:
    yoga_pose_path = os.path.join(train_test_folder, yoga_pose)
    if os.path.isdir(yoga_pose_path):
        frames_list = [os.path.join(yoga_pose_path, frame) for frame in os.listdir(yoga_pose_path)]

        # 80% training, 20% testing
        train_frames, test_frames = train_test_split(frames_list, test_size=0.2, random_state=42)

        train_pose_folder = os.path.join(train_folder, yoga_pose)
        test_pose_folder = os.path.join(test_folder, yoga_pose)

        os.makedirs(train_pose_folder)
        os.makedirs(test_pose_folder)

        for frame in train_frames:
            shutil.move(frame, train_pose_folder)

        for frame in test_frames:
            shutil.move(frame, test_pose_folder)