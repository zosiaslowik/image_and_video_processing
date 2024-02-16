import cv2
import os

def extract_frames(video_path, output_folder, max_frames=25):
    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return frames

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Limit the number of frames to the specified maximum
    max_frames = min(max_frames, frame_count)

    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(output_folder, f"frame_{i:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

        frames.append(frame)

    cap.release()
    return frames

def extract_frames_from_directory(directory_path, output_root_folder, max_frames_per_video=25):
    for video_file in os.listdir(directory_path):
        if video_file.endswith(".mp4"):  # Adjust the condition based on your video format
            video_path = os.path.join(directory_path, video_file)
            output_folder = os.path.join(output_root_folder, os.path.splitext(video_file)[0])  # Use video file name as the output folder name
            frames = extract_frames(video_path, output_folder, max_frames=max_frames_per_video)

            # Display or save some frames for visualization
            for i in range(min(5, len(frames))):  # Display the first 5 frames
                cv2.imshow(f"Frame {i+1} - {video_file}", frames[i])
                cv2.waitKey(1000)  # Wait for 1 second between frames

            cv2.destroyAllWindows()

# chair pose
directory_path_chair = '/Users/natalia_pyzara/Desktop/dataset_yoga/chair'
output_root_folder_chair = "/Users/natalia_pyzara/Desktop/output_folder_chair"

extract_frames_from_directory(directory_path_chair, output_root_folder_chair, max_frames_per_video=25)

# dolphin pose
directory_path_dolphin = '/Users/natalia_pyzara/Desktop/dataset_yoga/dolphin'
output_root_folder_dolphin = "/Users/natalia_pyzara/Desktop/output_folder_dolphin"

extract_frames_from_directory(directory_path_dolphin, output_root_folder_dolphin, max_frames_per_video=25)

# tree pose
directory_path_tree = '/Users/natalia_pyzara/Desktop/dataset_yoga/tree'
output_root_folder_tree = "/Users/natalia_pyzara/Desktop/output_folder_tree"

extract_frames_from_directory(directory_path_tree, output_root_folder_tree, max_frames_per_video=25)

# swanasana pose
directory_path_svanasana = '/Users/natalia_pyzara/Desktop/dataset_yoga/svanasana'
output_root_folder_svanasana = "/Users/natalia_pyzara/Desktop/output_folder_svanasana"

extract_frames_from_directory(directory_path_svanasana, output_root_folder_svanasana, max_frames_per_video=25)

# warrior_2 pose
directory_path_warrior_2 = '/Users/natalia_pyzara/Desktop/dataset_yoga/warrior_2'
output_root_folder_warrior_2 = "/Users/natalia_pyzara/Desktop/output_folder_warrior_2"

extract_frames_from_directory(directory_path_warrior_2, output_root_folder_warrior_2, max_frames_per_video=25)

# wheel_pose
directory_path_wheel_pose = '/Users/natalia_pyzara/Desktop/dataset_yoga/wheel_pose'
output_root_folder_wheel_pose = "/Users/natalia_pyzara/Desktop/output_folder_wheel_pose"

extract_frames_from_directory(directory_path_wheel_pose, output_root_folder_wheel_pose, max_frames_per_video=25)