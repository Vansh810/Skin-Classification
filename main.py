import numpy as np
import cv2
import os
from glob import glob
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split
from tkinter import Tk, filedialog, Label, Button, Canvas
from PIL import Image, ImageTk
import tensorflow as tf

# Define the data path and disease labels
DATA_PATH = 'Dataset'
PROCESSED_IMAGES = 'images.npy'
PROCESSED_LABELS = 'labels.npy'
OUTPUT_LABELS = ['acne', 'clear', 'wrinkle']
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 25

# Global variables for GUI
selected_image_path = None


def center_crop(image, target_size):
    """Center crop the image to the smallest dimension and then resize to target_size"""
    h, w, _ = image.shape
    crop_size = min(h, w)
    if h != w:
        image = tf.image.central_crop(image, central_fraction=crop_size / max(h, w))
    image = tf.image.resize(image, target_size)
    return image


def preprocess():
    try:
        # Load preprocessed images and labels if they exist
        images = np.load(PROCESSED_IMAGES)
        labels = np.load(PROCESSED_LABELS)
        print("Loaded Preprocessed Files")
    except FileNotFoundError:
        # Read and preprocess images if preprocessed files are not found
        images = []
        labels = []
        for label in OUTPUT_LABELS:
            img_files = glob(os.path.join(DATA_PATH, label, '*.jpg'))
            for file in img_files:
                img = cv2.imread(file)
                img = center_crop(img, IMG_SIZE)  # Apply center crop and resize
                images.append(img)
                labels.append(OUTPUT_LABELS.index(label))

        images = np.array(images)
        labels = np.array(labels)

        # Normalize the pixel values
        images = images / 255.0

        # Save the preprocessed images and labels
        np.save(PROCESSED_IMAGES, images)
        np.save(PROCESSED_LABELS, labels)

        print("Preprocessing Done")
    finally:
        # Split the data into training and testing sets
        return train_test_split(images, labels, test_size=0.2)


def evaluate(model, X_test, y_test):
    # Evaluate the model on the test data and print the accuracy.
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Accuracy: {accuracy * 100:.2f}%')


def train_model(X_train, X_test, y_train, y_test):
    try:
        # Load saved model if available
        model = load_model('skin_classification.keras')
        print("Loaded Saved Model")
    except (ValueError, FileNotFoundError):
        # Create a new model
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            Dropout(0.5),
            Dense(len(OUTPUT_LABELS), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Data augmentation for training images
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
        )
        datagen.fit(X_train)

        # Add EarlyStopping and ModelCheckpoint callbacks
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
        checkpoint = ModelCheckpoint('best_skin_classification.keras', monitor='val_accuracy', save_best_only=True)

        # Train the model
        model.fit(
            datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
            epochs=EPOCHS,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, checkpoint]
        )

        # Save the final model
        model.save('skin_classification.keras')
        print("Model Trained")

    return model


def identify_disease(image_path, model):
    # Load the image using OpenCV
    img = cv2.imread(image_path)
    img = center_crop(img, IMG_SIZE)  # Apply center crop and resize
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    disease_index = np.argmax(prediction)
    output = OUTPUT_LABELS[disease_index]
    return output, prediction[0][disease_index]


def select_image():
    global selected_image_path, canvas, img_display
    selected_image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if selected_image_path:
        # Display the selected image on the Canvas
        img = Image.open(selected_image_path)
        img = img.resize((302, 302))  # Resize for preview
        img_display = ImageTk.PhotoImage(img)
        canvas.create_image(151, 151, anchor='center', image=img_display)


def scan_image():
    global selected_image_path, output_label
    if selected_image_path:
        disease_label, confidence = identify_disease(selected_image_path, model)
        output_label.config(text=f"Status: {disease_label}\nConfidence: {confidence:.2f}")
    else:
        output_label.config(text="Please select an image first!")


# Preprocess the data and train the model
X_train, X_test, y_train, y_test = preprocess()
model = train_model(X_train, X_test, y_train, y_test)
evaluate(model, X_test, y_test)

# Create the GUI
root = Tk()
root.title("Skin Disease Classification")
root.geometry("600x600")

# Image preview area using Canvas
canvas = Canvas(root, width=300, height=300, bd=2, relief='groove')
canvas.pack(pady=20)

# Select Image button
select_btn = Button(root, text="Select Image", command=select_image)
select_btn.pack(pady=10)

# Scan button
scan_btn = Button(root, text="Scan", command=scan_image)
scan_btn.pack(pady=10)

# Output label
output_label = Label(root, text="", bg="white", width=40, height=2, wraplength=300)
output_label.pack(pady=20)

# Start the GUI loop
root.mainloop()
