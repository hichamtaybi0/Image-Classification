# Image-Classification
Features based image classification using machine learning algorithms

Target variable is either obj_car or obj_ship (binary classification)

Features extractes are: Hu Moments, Haralick Texture and Color Histogram

set folder that holds train images train_path = ""
if images in different size, remove comment from #fixed_size = tuple((120, 80)) and set your size, 
and from #image = cv2.resize(image, fixed_size) 

Features extracted will be stored in "Extracted_Features/data.h5"
