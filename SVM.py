import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import csv
from sklearn.metrics import classification_report, confusion_matrix

train_data = pd.read_csv('/kaggle/input/unibuc-dhc-2023/train.csv')
val_data = pd.read_csv('/kaggle/input/unibuc-dhc-2023/val.csv')
test_data = pd.read_csv('/kaggle/input/unibuc-dhc-2023/test.csv')

train_images = train_data['Image'].tolist()
train_labels = [int(label) for label in train_data['Class']]

val_images = val_data['Image'].tolist()
val_labels = [int(label) for label in val_data['Class']]

test_images = test_data['Image'].tolist()

def preprocess_train(photo):
    path = '/kaggle/input/unibuc-dhc-2023/train_images/' + str(photo)
    image = plt.imread(path)
    return image.reshape(-1)

def preprocess_val(photo):
    path = '/kaggle/input/unibuc-dhc-2023/val_images/' + str(photo)
    image = plt.imread(path)
    return image.reshape(-1)

def preprocess_test(photo):
    path = '/kaggle/input/unibuc-dhc-2023/test_images/' + str(photo)
    image = plt.imread(path)
    return image.reshape(-1)

train_features = np.array([preprocess_train(image) for image in train_images])
train_features = (train_features * 255).astype(int)

val_features = np.array([preprocess_val(image) for image in val_images])
val_features = (val_features * 255).astype(int)

C = 5
model = SVC(kernel = 'linear', C = C)
model.fit(train_features, train_labels)

val_predictions = model.predict(val_features)
accuracy = np.mean(val_predictions == val_labels)
print("Accuracy: ", accuracy)

classification_report = classification_report(val_labels, val_predictions, zero_division = 0)
print(classification_report)
confusion_matrix = confusion_matrix(val_labels, val_predictions)
print(confusion_matrix)

test_features = np.array([preprocess_test(image) for image in test_images])
test_features = (test_features * 255).astype(int)

test_predictions = model.predict(test_features)

submission_file = 'sample_submission3.csv'

with open(submission_file, 'w', newline = '') as document:
    writer = csv.writer(document)
    writer.writerow(['Image', 'Class'])
    for image, prediction in zip(test_images, test_predictions):
        writer.writerow([image, prediction])