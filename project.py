# Import the required modules
import cv2
import os
import numpy as np
import random
from sklearn.model_selection import KFold

# Define the path to the dataset folder
dataset_path = "./dataset/"

# Define the labels for the athletes
labels = os.listdir(dataset_path)
#print(labels)

# Define the face detector and the face recognizer
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Define a function to split the dataset into train and test data
def split_dataset():
    # Create empty lists to store the train and test data
    train_data = []
    test_data = []
    train_labels = []
    test_labels = []

    # Loop through each label
    for i, label in enumerate(labels):
        # Get the path to the label folder
        label_path = os.path.join(dataset_path, label)
        # Get the list of image files in the label folder
        image_files = os.listdir(label_path)
        # Shuffle the image files
        random.shuffle(image_files)
        # Split the image files into 80% train and 20% test
        train_size = int(len(image_files) * 0.8)
        train_files = image_files[:train_size]
        test_files = image_files[train_size:]
        # Loop through the train files
        for train_file in train_files:
            # Get the full path to the image file
            image_path = os.path.join(label_path, train_file)
            # Read the image as grayscale
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # Define the kernel size and standard deviation
            kernel_size = 9
            sigma = 4.5
            # Apply the Gaussian blur filter to the image
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
            # Detect the face in the image
            faces = face_detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=3)
            # If exactly one face is detected, append the image and the label to the train data
            if len(faces) == 1:
                x, y, w, h = faces[0]
                face = image[y:y+h, x:x+w]
                train_data.append(face)
                train_labels.append(i)
        # Loop through the test files
        for test_file in test_files:
            # Get the full path to the image file
            image_path = os.path.join(label_path, test_file)
            # Read the image as grayscale
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # Define the kernel size and standard deviation
            kernel_size = 9
            sigma = 4.5
            # Apply the Gaussian blur filter to the image
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
            # Detect the face in the image
            faces = face_detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=3)
            # If exactly one face is detected, append the image and the label to the test data
            if len(faces) == 1:
                x, y, w, h = faces[0]
                face = image[y:y+h, x:x+w]
                test_data.append(face)
                test_labels.append(i)

    # Return the train and test data as numpy arrays
    return np.array(train_data), np.array(test_data), np.array(train_labels), np.array(test_labels)

# Define a function to train and test the model
def train_and_test_model():
    # Split the dataset into train and test data
    train_data, test_data, train_labels, test_labels = split_dataset()
    # Concatenate the train and test data and labels
    data = np.concatenate((train_data, test_data))
    labels = np.concatenate((train_labels, test_labels))
    # Define the number of folds
    k = 5
    # Create a KFold object
    kfold = KFold(n_splits=k, shuffle=True, random_state=72)
    # Initialize a list to store the accuracy scores
    scores = []
    # Loop through the k folds
    for train_index, test_index in kfold.split(data):
        # Get the train and test data and labels for the current fold
        train_data = data[train_index]
        test_data = data[test_index]
        train_labels = labels[train_index]
        test_labels = labels[test_index]
        # Train the face recognizer on the train data
        face_recognizer.train(train_data, train_labels)
        # Save the model to a file
        face_recognizer.write("model.yml")
        # Initialize a variable to count the number of correct predictions
        correct = 0
        # Loop through the test data
        for i, face in enumerate(test_data):
            # Predict the label of the face
            label, confidence = face_recognizer.predict(face)
        # Calculate and append the accuracy score of the prediction
        score = (1-(confidence / 300)) *100
        scores.append(score)
    # Calculate and print the average accuracy score
    accuracy = np.mean(scores)
    print(f"Average accuracy: {accuracy:.2f}%")

# Define a function to predict a new image
def predict():
    # Load the model from the file
    face_recognizer.read("model.yml")
    # Ask the user to input the image path
    image_path = input("Enter the image path: ")
    # Read the image as color
    image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Define the kernel size and standard deviation
    kernel_size = 9
    sigma = 4.5
    # Apply the Gaussian blur filter to the image
    image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    # Detect the faces in the image
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    # Loop through the faces
    for x, y, w, h in faces:
        # Crop the face from the image
        face = gray[y:y+h, x:x+w]
        # Predict the label of the face
        label, confidence = face_recognizer.predict(face)
        confidence = (1-(confidence / 300)) *100
        # Draw a rectangle around the face
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Put the label and the confidence on the image
        cv2.putText(image, labels[label], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(image, f"{confidence:.2f}", (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    # Show the image
    cv2.imshow("Image", image)
    # Wait for a key press to close the image
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Define the main function
def main():
    # Define a flag to control the loop
    flag = True
    # Loop until the user chooses to exit
    while flag:
        # Print the menu options
        print("Menu:")
        print("1. Train and test model")
        print("2. Predict")
        print("3. Exit")
        # Ask the user to choose an option
        choice = input("Enter your choice: ")
        # If the user chooses option 1, call the train and test function
        if choice == "1":
            train_and_test_model()
        # If the user chooses option 2, call the predict function
        elif choice == "2":
            predict()
        # If the user chooses option 3, set the flag to false
        elif choice == "3":
            flag = False
        # If the user chooses an invalid option, print an error message
        else:
            print("Invalid choice. Please try again.")

# Call the main function
if __name__ == "__main__":
    main()
