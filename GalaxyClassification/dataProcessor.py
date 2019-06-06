import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
import skimage.transform as sktransform

def_path = '/Users/vshrivas/workspace/Vaishnavi/Caltech/CS/AstroML'

def get_image_paths(image_dir, numImages):
    directory = def_path + image_dir
    return [file for file in os.listdir(directory)][0: numImages]

"""
Find image paths, import image at 'paths', decode, centre crop, and store in array.
"""
def process_images(image_dir, numImages):
    directory = def_path + image_dir
    paths = [file for file in os.listdir(directory)][0: numImages]

    count = numImages
    arr = np.zeros(shape=(count,106,106, 3))
    for c, path in enumerate(paths):
        img = plt.imread(directory + '/' + path).T
        img = img[:,106:106*3,106:106*3] #crop 424x424 -> 212x212
        img = sktransform.resize(img, (106,106,3))
        #img = imresize(img,size=(106,106,3),interp="cubic").T # downsample to half res
        arr[c] = img
        #print(img)
    return arr

def process_labels(labelsFile, nPoints):
    df_data_labels = pd.read_csv(def_path + labelsFile,nrows = nPoints)
    df_data_labels['Max'] = df_data_labels[df_data_labels.columns.difference(['GalaxyID'])].idxmax(axis=1)

    #print (df_data_labels.columns.difference(['GalaxyID', 'Max']))
    labelsArray = np.zeros((nPoints, 37))
    for index, row in df_data_labels.iterrows():
        class_prediction = row['Max']
        class_index = df_data_labels.columns.difference(['GalaxyID', 'Max']).get_loc(class_prediction)
        labelsArray[index][class_index] = 1

    return labelsArray

def dp_main(nPoints):
    image_dir = '/datasets/images_training_rev1'
    labels_file = '/datasets/training_solutions_rev1.csv'

    print("processing labels...")
    labelsArray = process_labels(labels_file, nPoints)
    print("processing images...")
    imgsArray = process_images(image_dir, nPoints)

    return (imgsArray, labelsArray)
