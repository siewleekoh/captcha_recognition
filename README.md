# Captcha Recognition API

### Description of the project
This is a Python Flask-based API built to recognize captcha. Python Tesseract could not solve the captcha on the website and a neural network model is developed to address the issue.

This API server is currently hosted at localhost.

The model is built using ~5000 manually labelled captchas. The extracted letters from the captchas were augmented to rotate at ~40 degree randomly to accommodate the changing captcha. Hyperparameter tuning of the neural network model were also done.
The accuracy of the model is at ~0.99 for the testing dataset. Running the model on validation dataset (with ~2000 captchas) gives ~0.9 accuracy.

Detailed methods are outlined in the Jupyter Notebook. The model is stored in the 'model' folder. Refer to notes below should there be a need to retrain the model.

&nbsp;

**Final model**  


|Layer (type)|  Output Shape    |           Param #   |
|---|---|---|
|conv2d (Conv2D)    |          (None, 26, 26, 84)   |     840       |
|conv2d_1 (Conv2D)    |        (None, 24, 24, 224)  |     169568   |
|max_pooling2d (MaxPooling2D) |(None, 12, 12, 224)   |    0     |    
|flatten (Flatten)         |   (None, 32256)       |      0     |    
|dense (Dense)    |            (None, 564)       |        18192948 |
|dropout (Dropout)|            (None, 564)    |           0    |     
|dense_1 (Dense) |             (None, 10)           |     5650   |   

**Total params: 18,369,006  
Trainable params: 18,369,006  
Non-trainable params: 0**



&nbsp;


### Steps to train new model (in local drive)

**a. Open cmd prompt, change to working directory and create a new virtual environment if you have not already**  

```
cd /user/xxxx/captcha_recognition/
```

```
pip install -r requirements.txt
```


**b. Download captcha images**
- The script uses Selenium to open a Firefox browser at the back and download captchas from the website. The images will be stored in the 'images' folder.  Scripts has to be modified accordingly to accommodate captcha format from different websites.   

```
python download_captcha.py
```

**c. Manual labelling of downloaded captcha images**  
- You need to now manually label captcha images for the training of the model. Currently, ~4000 images are labelled.

**d. Image Processing**  
- Extract individual letters from the labelled 4-letter captcha and put in individual folder for the 10 numbers

```
python extract_letters_from_captcha.py
```

**e. Tune Keras model**  

```
python hyperparameter_tuning.py
```


**e. Train Keras model**  
- Modify model according after finding the best parameters and run the following
```
python model_training.py
```
