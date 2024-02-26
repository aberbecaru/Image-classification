import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

train_data = pd.read_csv('/kaggle/input/unibuc-dhc-2023/train.csv')
val_data = pd.read_csv('/kaggle/input/unibuc-dhc-2023/val.csv')
test_data = pd.read_csv('/kaggle/input/unibuc-dhc-2023/test.csv')

train_images = train_data['Image'].tolist()
train_labels = [int(label) for label in train_data['Class']]

val_images = val_data['Image'].tolist()
val_labels = [int(label) for label in val_data['Class']]

test_images = test_data['Image'].tolist()

def preprocess_image(photo, folder):
    path = '/kaggle/input/unibuc-dhc-2023/' + folder + str(photo)
    image = Image.open(path).convert('L')
    image = image.resize((64, 64))
    image = np.array(image)
    image = np.expand_dims(image, axis = -1)
    image = image / 255.0
    return image

def preprocess_data(images, labels, folder):
    features = np.array([preprocess_image(image, folder) for image in images])

    if labels is not None:
        labels = tf.keras.utils.to_categorical(labels, num_classes = 96)

    return features, labels

train_features, train_labels = preprocess_data(train_images, train_labels, 'train_images/')
val_features, val_labels = preprocess_data(val_images, val_labels, 'val_images/')
test_features, _ = preprocess_data(test_images, None, 'test_images/')

model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (train_features.shape[1], train_features.shape[2], 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation = 'relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation = 'relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation = 'relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation = 'relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation = 'relu'),
    layers.Dropout(0.5),
    layers.Dense(96, activation = 'softmax')
])

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(train_features, train_labels, epochs = 50, batch_size = 64, validation_data = (val_features, val_labels))

val_predictions = model.predict(val_features)
val_predicted_labels = np.argmax(val_predictions, axis = 1)
val_true_labels = np.argmax(val_labels, axis = 1)
accuracy = np.mean(val_predicted_labels == val_true_labels)
print("Accuracy: ", accuracy)

classification_report = classification_report(val_true_labels, val_predicted_labels)
print(classification_report)
confusion_matrix = confusion_matrix(val_true_labels, val_predicted_labels)
print(confusion_matrix)

test_predictions = model.predict(test_features)
test_labels = test_predictions.argmax(axis = 1)
test_labels = test_labels.astype(int)

submission_file = 'sample_submission9.csv'

with open(submission_file, 'w', newline = '') as document:
    writer = csv.writer(document)
    writer.writerow(['Image', 'Class'])
    for image, prediction in zip(test_images, test_predictions):
        writer.writerow([image, prediction])

