import cv2
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

dataset_path = "./dataset/"

labels = os.listdir(dataset_path)
#print(labels)

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

kernel_size = 9
sigma = 4.5

def split_dataset():
    train_data = []
    test_data = []
    train_labels = []
    test_labels = []

    for i, label in enumerate(labels):
        label_path = os.path.join(dataset_path, label)
        image_files = os.listdir(label_path)

        train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)
        for train_file in train_files:
            image_path = os.path.join(label_path, train_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            image = cv2.equalizeHist(image)
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            image = clahe.apply(image)

            faces = face_detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=3)
            if len(faces) == 1:
                x, y, w, h = faces[0]
                face = image[y:y+h, x:x+w]
                train_data.append(face)
                train_labels.append(i)

        for test_file in test_files:
            image_path = os.path.join(label_path, test_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            image = cv2.equalizeHist(image)
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            image = clahe.apply(image)

            faces = face_detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=3)
            if len(faces) == 1:
                x, y, w, h = faces[0]
                face = image[y:y+h, x:x+w]
                test_data.append(face)
                test_labels.append(i)

    return np.array(train_data), np.array(test_data), np.array(train_labels), np.array(test_labels)

k = 5
def train_and_test_model():
    train_data, test_data, train_labels, test_labels = split_dataset()

    data = np.concatenate((train_data, test_data))
    labels = np.concatenate((train_labels, test_labels))
    kfold = KFold(n_splits=k, shuffle=True, random_state=72)

    scores = []
    for train_index, test_index in kfold.split(data):
        train_data = data[train_index]
        test_data = data[test_index]
        train_labels = labels[train_index]
        test_labels = labels[test_index]

        face_recognizer.train(train_data, train_labels)
        face_recognizer.write("model.yml")

        pred_labels = []
        for i, face in enumerate(test_data):
            label, confidence = face_recognizer.predict(face)
            pred_labels.append(label)

        score = accuracy_score(test_labels, pred_labels)
        score = (1-(confidence / 300)) *100
        scores.append(score)

    accuracy = np.mean(scores)
    print(f"Accuracy: {accuracy:.2f}%")

def predict():
    face_recognizer.read("model.yml")
    image_path = input("Enter the image path: ")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    image = cv2.equalizeHist(image)
    image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image = clahe.apply(image)

    faces = face_detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=3)
    for x, y, w, h in faces:
        face = image[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(face)
        confidence = (1-(confidence / 300)) *100

        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, labels[label], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(image, f"{confidence:.2f}", (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    #cv2.imshow("Image", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    plt.imshow(image)
    plt.show()

def main():
    flag = True
    while flag:
        print("Menu:")
        print("1. Train and test model")
        print("2. Predict")
        print("3. Exit")
        choice = input("Enter your choice: ")
        if choice == "1":
            train_and_test_model()
        elif choice == "2":
            predict()
        elif choice == "3":
            flag = False
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
