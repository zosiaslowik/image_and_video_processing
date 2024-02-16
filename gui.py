import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image, ImageTk
import mediapipe as mp

# Load the first model
model_1 = load_model("yogaPoseClassifier.model")

# Define class labels for the first model
class_labels_1 = ["bhujasana", "padamasana", "tadasana", "trikasana", "vrikshasana"]

# Load the second model
model_2 = load_model('model_2')

# Define class labels for the second model
class_labels_2 = ["chair pose", "dolphin", "svansana", "tree pose", "warrior_2", "wheel pose"]

# MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Create a GUI window
root = tk.Tk()
root.title("Yoga Pose Classifier")
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Set the window size
root.geometry(f"{screen_width}x{screen_height}")
root.configure(bg="black")  # Set the background color to black

# Load the header image
header_image_path = "image.png"  # Replace with the actual path to your image
header_image = Image.open(header_image_path)
header_image_tk = ImageTk.PhotoImage(header_image)

# Display the header image
header_label = tk.Label(root, image=header_image_tk, bg="black")
header_label.pack()

# Create a variable to store the selected model
selected_model = tk.StringVar(value="model_1")

# Function to classify video frames based on the selected model
def classify_video(video_path):
    if selected_model.get() == "model_1":
        return classify_video_model_1(video_path)
    elif selected_model.get() == "model_2":
        return classify_video_model_2(video_path)

# Function to classify video frames using the first model
def classify_video_model_1(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_label.config(text='Uploading...', fg='white')
    root.update_idletasks()
    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Update progress bar
        progress_var.set(int((frame_num + 1) / total_frames * 100))
        progress_bar_label.config(text=f"{int((frame_num + 1) / total_frames * 100)}%")
        root.update_idletasks()

        # Resize frame to match the input size of the model
        img = cv2.resize(frame, (150, 150))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.0
        frames.append(img)

    cap.release()

    frames = np.vstack(frames)
    progress_bar.pack_forget()
    progress_bar_label.pack_forget()
    progress_label.config(text='Predicting Yoga Pose...', fg='white')  # Set text color to white

    root.update_idletasks()
    predictions = model_1.predict(frames, batch_size=10)

    # Get the majority-voted class for the entire video
    predicted_classes = np.argmax(predictions, axis=1)
    majority_voted_class = np.bincount(predicted_classes).argmax()
    predicted_label = class_labels_1[majority_voted_class]

    return predicted_label

# Function to preprocess video frames for model_2
def preprocess_frame_model_2(frame):
    # Process the frame to detect poses
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Check if poses are detected
    if results.pose_landmarks:
        landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.pose_landmarks.landmark]).flatten()
        return landmarks.reshape(1, -1)
    else:
        # Return an empty array if no poses are detected
        return np.array([])

# Function to classify video frames using the second model
# Function to classify video frames using the second model
def classify_video_model_2(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_label.config(text='Uploading...', fg='white')
    root.update_idletasks()
    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Update progress bar
        progress_var.set(int((frame_num + 1) / total_frames * 100))
        progress_bar_label.config(text=f"{int((frame_num + 1) / total_frames * 100)}%")
        root.update_idletasks()

        # Process the frame for model_2
        processed_frame = preprocess_frame_model_2(frame)

        if processed_frame.size > 0:
            frames.append(processed_frame)

    cap.release()

    if not frames:
        # Return an appropriate value when no poses are detected
        return "No poses detected"


    frames = np.vstack(frames)

    progress_label.config(text='Predicting Pose...', fg='white')  # Set text color to white
    root.update_idletasks()
    predictions = model_2.predict(frames, batch_size=10)

    # Get the majority-voted class for the entire video
    predicted_classes = np.argmax(predictions, axis=1)
    majority_voted_class = np.bincount(predicted_classes).argmax()
    predicted_label = class_labels_2[majority_voted_class]

    return predicted_label



# Function to handle video classification and display the result
def classify_video_and_display():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
    if not file_path:
        return

    # Show "Uploading..." label
    progress_label.pack(pady=5)

    # Show progress bar
    progress_var.set(0)
    progress_bar.pack(pady=10)
    progress_bar_label.config(text="0%")
    progress_bar_label.pack()

    predicted_label = classify_video(file_path)



    # Hide progress bar
    progress_bar.pack_forget()
    progress_bar_label.pack_forget()

    # Display the result in a label
    progress_label.config(text="Predicted Pose: " + predicted_label, fg='white')  # Set text color to white
    root.update_idletasks()

# Function to exit the application
def exit_application():
    root.destroy()

# Create and configure GUI elements
upload_button = tk.Button(root, text="Upload Video", command=classify_video_and_display, bg='white', fg='black', height=3, width=20, font=("Courier", 16))
upload_button.pack(pady=20)

# Model selection dropdown
model_selection_dropdown = tk.OptionMenu(root, selected_model, "model_1", "model_2")
model_selection_label = tk.Label(root, text="Select Model:", bg="black", fg="white", font=("Courier", 16))
model_selection_label.pack()
model_selection_dropdown.pack(pady=5)

# Exit button
exit_button = tk.Button(root, text="Exit", command=exit_application, bg='white', fg='black', height=3, width=20, font=("Courier", 16))
exit_button.pack(pady=10)

# Label for "Uploading..."
progress_label = tk.Label(root, text="Uploading...", bg="black", fg="white", font=("Courier", 20))

# Progress bar to show the progress of video uploading
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(root, variable=progress_var, length=300, mode="determinate", style="white.Horizontal.TProgressbar")
progress_bar_label = tk.Label(root, text="0%", bg="black", fg="white")

# Text label at the bottom
bottom_text_label = tk.Label(root, text="Pose predicted by model1: Bhujasana, Padamasana, Tadasana, Trikasana, "
                                        "Vrikshasana\nPose predicted by model2: chair, dolphin, svanasana, tree, warrior2, wheel", bg="black", fg="white", font=("Courier", 20))
bottom_text_label.pack(side="bottom", pady=80)

# Run the GUI application
root.mainloop()