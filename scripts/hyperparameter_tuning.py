from kerastuner.tuners import RandomSearch
import os.path
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from imutils import paths
import numpy as np
from PIL import Image
from helper_function import resize_to_fit


# function to tune hyperparameters
def build_model(hp):
    model = keras.models.Sequential()
    model.add(Conv2D(filters=hp.Int('input_units', min_value=20, max_value=320, step=16),
                               kernel_size = (3,3),
                               input_shape =(28, 28, 1),
                               activation = "relu"))

    
    for i in range(hp.Int("n_layers", 1, 2)):
        model.add(Conv2D(filters=hp.Int(f"conv_{i}_units", min_value=32, max_value=256, step=32),
                           kernel_size= (3,3),
                           activation="relu"))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    
    model.add(Flatten())
    
    model.add(Dense(units=hp.Int('dense_1_units', min_value=20, max_value=640, step=32), activation='relu'))
    
    model.add(Dropout(hp.Float('dropout',min_value=0,max_value=0.5,step=0.1)))
    
    model.add(Dense(10, activation='softmax'))
    
    model.compile(
    optimizer = keras.optimizers.Adam(hp.Float('learning_rate',
                                            min_value=1e-4,
                                            max_value=1e-2,
                                            sampling='LOG',
                                            default=1e-3
                                        )
                                    ),
                                    loss=keras.losses.categorical_crossentropy,
                                    metrics=['accuracy']
                                )

    return model



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
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.3, random_state=12)
print('======X_train.shape', X_train.shape)
print('======X_test.shape', X_test.shape)
print('======Y_train.shape', Y_train.shape)
print('======Y_test.shape', Y_test.shape)

# Convert the labels (letters) into one-hot encodings that Keras can work with
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)



tuner = RandomSearch(
         build_model,
         objective='val_acc',
         max_trials=40,
         executions_per_trial=1,
        directory=os.path.normpath('C:/Users/kohsi/Desktop/'),
        project_name='captcha_20200409')


tuner.search(X_train, Y_train, epochs=10, validation_data=(X_test, Y_test))