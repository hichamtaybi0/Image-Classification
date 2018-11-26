# Image-Classification
Features based image classification using machine learning algorithms

Target variable is either obj_car or obj_ship (binary classification)

Features extractes are: Hu Moments, Haralick Texture and Color Histogram

Separate images to be each class in folder named as target varibale values (ex: obj_car) or download it from here  https://drive.google.com/open?id=1BVe_me1ZCVoeCtZ1UWPGxnhRQj5X_CAo

In features.py set Parent directory into train_path = ""

if images in different size, remove comment from #fixed_size = tuple((120, 80)) and set your size, 
and from #image = cv2.resize(image, fixed_size) 

run features.py, Features extracted will be stored in "Extracted_Features/data.h5" and labels in "Extracted_Features/labels.h5"

run main.py to see how well work different classification algorithms and the performance on train/test Data

run test.py to predict (Classify) new images in DataToPredict folder using RandomForest Algorithm, train the model on the stored examples in Extracted_Features. extract features from images in DataToPredict by the same methods used for training images.
