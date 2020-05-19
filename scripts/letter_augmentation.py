import os.path
import glob
from PIL import Image
import numpy as np
import tensorflow as tf


# working directory
captcha_home_path = os.path.abspath(__file__ + "/../../")
INPUT_IMAGES_FOLDER = '/images/extracted_letters/extracted_letter_images_left/'
OUTPUT_IMAGES_FOLDER = '/images/extracted_letters/extracted_letter_images_left_augmented/'


# Get a list of all the captcha images we need to process
for i in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
    ind_letter_files = glob.glob(captcha_home_path + INPUT_IMAGES_FOLDER + i + "/*")

    for j in range(len(ind_letter_files)):
        # single image
        print("processing folder ", i, " : image", j)
        img = Image.open(ind_letter_files[j])
        img_array = np.array(img)
        img_array = img_array.reshape(40, 40, 1)

        for iteration in range(10): #augment each image 10 times
            img_rotated_array = tf.keras.preprocessing.image.random_rotation(
                img_array,
                rg=40,
                row_axis=1,
                col_axis=0,
                channel_axis=2,
                fill_mode='nearest',
                cval=0)
            new_image_array = img_rotated_array.reshape(40, 40)
            new_image = Image.fromarray(new_image_array)  # convert np.array back to image

            # Get the folder to save the image in
            letter_text = i
            save_path = os.path.join(captcha_home_path + OUTPUT_IMAGES_FOLDER, letter_text)

            # if the output directory does not exist, create it
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            try:
                max_fileno = max([int(f[:f.index('.')]) for f in os.listdir(save_path)])
            except ValueError:
                max_fileno = 0

            p = os.path.join(save_path, "{}.png".format(str(max_fileno + 1).zfill(6)))
            print(p)

            new_image.save(p)
