from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import cv2
import os
import random

def extract_random_frames(video_path, output_folder, max_frames=35):
    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return frames

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frame_indices = random.sample(range(frame_count), min(max_frames, frame_count))

    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(output_folder, f"frame_{i:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

        frames.append(frame)

    cap.release()
    return frames

def extract_random_frames_from_directory(directory_path, output_root_folder, max_frames=35):
    for folder_name in os.listdir(directory_path):
        folder_path = os.path.join(directory_path, folder_name)

        if os.path.isdir(folder_path):
            output_folder = os.path.join(output_root_folder, folder_name)

            for video_file in os.listdir(folder_path):
                if video_file.endswith(".mp4"):
                    video_path = os.path.join(folder_path, video_file)

                    frames = extract_random_frames(video_path, output_folder, max_frames)


directory_path_yoga_dataset = 'dataset/'
output_root_folder_yoga_dataset = "split_dataset/"
dirs = ['train', 'test', 'val']
for dir in dirs:
    extract_random_frames_from_directory(directory_path_yoga_dataset+dir, output_root_folder_yoga_dataset+dir, max_frames=35)

train_dir = 'split_dataset/train'

validation_dir = 'split_dataset/val'

test_dir = 'split_dataset/test'


train_bhujasana_dir = os.path.join(train_dir, "bhujasana")
train_padamasana_dir = os.path.join(train_dir, "padamasana")
train_tadasana_dir = os.path.join(train_dir, "tadasana")
train_trikasana_dir = os.path.join(train_dir, "trikasana")
train_vrikshasana_dir = os.path.join(train_dir, "vrikshasana")


validation_bhujasana_dir = os.path.join(validation_dir, "bhujasana")
validation_padamasana_dir = os.path.join(validation_dir, "padamasana")
validation_tadasana_dir = os.path.join(validation_dir, "tadasana")
validation_trikasana_dir = os.path.join(validation_dir, "trikasana")
validation_vrikshasana_dir = os.path.join(validation_dir, "vrikshasana")


test_bhujasana_dir = os.path.join(test_dir, "bhujasana")
test_padamasana_dir = os.path.join(test_dir, "padamasana")
test_tadasana_dir = os.path.join(test_dir, "tadasana")
test_trikasana_dir = os.path.join(test_dir, "trikasana")
test_vrikshasana_dir = os.path.join(test_dir, "vrikshasana")


model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation="relu", input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(5, activation="softmax"))
model.summary()


model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["acc"],
)


train_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=25,
    class_mode="categorical",
    shuffle=True,
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir, target_size=(150, 150), batch_size=20, class_mode="categorical"
)


history = model.fit(
    train_generator,
    steps_per_epoch=30,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=10,
)


for data_batch, labels_batch in train_generator:
    print("data batch shape:", data_batch.shape)
    print("labels batch shape:", labels_batch.shape)
    break

acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(len(acc))

for data_batch, labels_batch in train_generator:
    print("data batch shape:", data_batch.shape)
    print("labels batch shape:", labels_batch.shape)
    break

model.save("yogaPoseClassifier.model")
