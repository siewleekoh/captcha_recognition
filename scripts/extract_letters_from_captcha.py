import os.path
import glob
from PIL import Image
import numpy as np
from helper_function import img_process


# working directory
captcha_home_path = os.path.abspath(__file__ + "/../../")
img_folders_path = captcha_home_path + '/images/labeled/labeled_training/'
img_folders = glob.glob(img_folders_path + "*")
OUTPUT_FOLDER = "/images/extracted_letters"


for folder in img_folders:

    # Get a list of all the captcha images we need to process
    captcha_image_files = glob.glob(folder + "\\*")

    # loop over the image paths
    for (i, captcha_image_file) in enumerate(captcha_image_files):
        print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))

        # Since the filename contains the captcha text (i.e. "0186.jpg" has the text "0.186"),
        # grab the base filename as the text
        filename = os.path.basename(captcha_image_file)
        captcha_correct_text = os.path.splitext(filename)[0]

        # Load the image and convert it to grayscale
        image = Image.open(captcha_image_file).convert("L")  # Grayscale conversion

        cropped_image1 = np.array(image.crop((10, 0, 50, 70)))
        cropped_image2 = np.array(image.crop((45, 0, 85, 70)))
        cropped_image3 = np.array(image.crop((80, 0, 120, 70)))
        cropped_image4 = np.array(image.crop((120, 0, 160, 70)))
        cropped_image = [cropped_image1, cropped_image2, cropped_image3, cropped_image4]

        # Save out each letter as a single image
        for ind_img, letter_text in zip(cropped_image, captcha_correct_text):

            img_array = img_process(ind_img, 40, 40)
            img = Image.fromarray(img_array)  # convert np.array back to image

            # Get the folder to save the image in
            save_path = os.path.join(captcha_home_path + OUTPUT_FOLDER, letter_text)

            # if the output directory does not exist, create it
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            try:
                max_fileno = max([int(f[:f.index('.')]) for f in os.listdir(save_path)])
            except ValueError:
                max_fileno = 0

            p = os.path.join(save_path, "{}.png".format(str(max_fileno + 1).zfill(6)))
            img.save(p)


