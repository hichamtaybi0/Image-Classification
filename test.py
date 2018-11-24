# import the necessary packages
import h5py
import numpy as np
import glob
import cv2
import mahotas
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import os
import csv
import ntpath
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


bins = 8
# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()


# import the feature vector and trained labels
h5f_data = h5py.File('Extracted_Features/data.h5', 'r')
h5f_label = h5py.File('Extracted_Features/labels.h5', 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()

'''
# split the training and testing data
(X_train, X_test, y_train, y_test) = train_test_split(np.array(global_features),
                                                      np.array(global_labels),
                                                      test_size=0.2,
                                                      random_state=9)
'''
X_train = np.array(global_features)
y_train = np.array(global_labels)



# create the model - Random Forests
rfc = RandomForestClassifier(n_estimators=4, random_state=9)

# fit the training data to the model
rfc.fit(X_train, y_train)

# path to test data
test_folder = 'DataToPredict'
test_path = "C:/Users/hicham/Desktop/S3/ImageMining/atel2"
path = os.path.join(test_path, test_folder)

with open('file.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',')
    filewriter.writerow(['Name', 'Classe'])

    # loop through the test images
    for file in glob.glob(os.path.join(path, '*.jpg')):
        # read the image
        image = cv2.imread(file)

        fv_hu_moments = fd_hu_moments(image)
        fv_haralick = fd_haralick(image)
        fv_histogram = fd_histogram(image)

        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

        # predict label of test image
        prediction = rfc.predict(global_feature.reshape(1, -1))[0]

        #extract image name from the path
        image_name = ntpath.basename(file)
        if(prediction == 0):

            filewriter.writerow([image_name, 'obj_car'])

        else:
            filewriter.writerow([image_name, 'obj_ship'])

        # show predicted label on image
        #cv2.putText(image, train_labels[prediction], (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        # display the output image
        #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        #plt.show()

