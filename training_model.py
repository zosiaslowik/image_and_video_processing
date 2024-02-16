import os
import numpy as np
from keras import layers, models
from sklearn.preprocessing import LabelEncoder


train_pose_data_dir = '/Users/natalia_pyzara/Desktop/train_pose_data_dir'
test_pose_data_dir = '/Users/natalia_pyzara/Desktop/test_pose_data_dir'


train_data = []
train_labels = []

for file in os.listdir(train_pose_data_dir):
    if file.endswith('.npy'):
        filepath = os.path.join(train_pose_data_dir, file)
        landmarks = np.load(filepath)
        label = file.split('_')[2]
        train_data.append(landmarks)
        train_labels.append(label)


test_data = []
test_labels = []

for file in os.listdir(test_pose_data_dir):
    if file.endswith('.npy'):
        filepath = os.path.join(test_pose_data_dir, file)
        landmarks = np.load(filepath)
        label = file.split('_')[2]
        test_data.append(landmarks)
        test_labels.append(label)


train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)

# label encoding
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# classification model
model = models.Sequential([
    layers.Flatten(input_shape=(train_data.shape[1],)),  # landmarks are flatten
    layers.Dense(128, activation='relu'),
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train the model
model.fit(train_data, train_labels_encoded, epochs=45, validation_data=(test_data, test_labels_encoded))

# evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_data, test_labels_encoded)
print(f'Test accuracy: {test_acc}')

model.save('model_2')
