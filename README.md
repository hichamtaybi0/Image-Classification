# Image-Classification
Features based image classification using machine learning algorithms

Target variable is either obj_car or obj_ship (binary classification)

Features extractes are: Hu Moments, Haralick Texture and Color Histogram

Separate images to be each class in folder named as target varibale values (ex: obj_car)

In features.py set Parent directory into train_path = ""

if images in different size, remove comment from #fixed_size = tuple((120, 80)) and set your size, 
and from #image = cv2.resize(image, fixed_size) 

run features.py, Features extracted will be stored in "Extracted_Features/data.h5" and labels in "Extracted_Features/labels.h5"
