from PIL import Image
import pickle
import os.path
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, Dropout
from helper_function import resize_to_fit, img_process



# working directory
captcha_home_path = os.path.abspath(__file__ + "/../../")
# folder names
LETTER_IMAGES_FOLDER =  [captcha_home_path + "/images/extracted_letters/extracted_letter_images_left_augmented",
                        captcha_home_path + "/images/extracted_letters/extracted_letter_images_right_augmented",
                         captcha_home_path + "/images/extracted_letters/extracted_letter_images_straight_augmented"]


MODEL_FILENAME = captcha_home_path + "/model/captcha_model.hdf5"
MODEL_LABELS_FILENAME = captcha_home_path + "/model/model_labels.dat"

# initialize the data and labels
data = []
labels = []

# loop over 2 folders
for image_folder in LETTER_IMAGES_FOLDER:
    folder = image_folder
    # loop over the input images
    for image_file in paths.list_images(folder):
        # Load the image and convert it to grayscale
        image = np.array(Image.open(image_file).convert("L"))  # Grayscale conversion
        # Resize the letter so it fits in a 28x28 pixel box
        image = resize_to_fit(image, 28, 28)

        # Add a third channel dimension to the image to make Keras happy
        image = np.expand_dims(image, axis=2)

        # Grab the name of the letter based on the folder it was in
        label = image_file.split(os.path.sep)[-2]

        # Add the letter image and it's label to our training data
        data.append(image)
        labels.append(label)

# scale the raw pixel intensities to the range [0, 1] (this improves training)
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Split the training data into separate train and test sets
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)
print('======X_train.shape', X_train.shape)
print('======X_test.shape', X_test.shape)
print('======Y_train.shape', Y_train.shape)
print('======Y_test.shape', Y_test.shape)

# Convert the labels (letters) into one-hot encodings that Keras can work with
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# Save the mapping from labels to one-hot encodings.
# We'll need this later when we use the model to decode what it's predictions mean
with open(MODEL_LABELS_FILENAME, "wb") as f:
    pickle.dump(lb, f)


# build model
model = keras.models.Sequential()
model.add(Conv2D(84, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=(28, 28, 1)))

model.add(Conv2D(224, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(564, activation='relu'))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(learning_rate= 0.00069425),
                loss=keras.losses.categorical_crossentropy,
                metrics=['accuracy'])



callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
# This callback will stop the training when there is no improvement in the validation loss for three consecutive epochs.
model.fit(X_train, Y_train,validation_data=(X_test, Y_test), batch_size=64,
          epochs=100, callbacks=[callback],verbose=1 )

# Save the trained model to disk
model.save(MODEL_FILENAME)