# Import Libraries

import pandas as pd
import numpy as np
from keras.preprocessing import image
from ImageFeatureExtractor import FeatureExtractor
image.LOAD_TRUNCATED_IMAGES = True
from DataPath import *
import os

def extractImageName(x):
    # 1. Invert the image path
    x_inv = x[:: -1]

    # 2. Find the index of '/'
    slash_idx = x_inv.find('/')

    # 3. Extract the text after the -slash_idx
    return x[-slash_idx:]

if __name__ == '__main__':

    # Load the data
    listing_data = pd.read_csv(listing_data)

    # Drop priceInfo.installmentsLabel
    listing_data.drop('priceInfo.installmentsLabel', axis=1, inplace=True)

    # Drop the column merchandiseLabel
    listing_data.drop('merchandiseLabel', axis=1, inplace=True)

    # fill the null values in priceInfo.discountLabel with 0
    listing_data['priceInfo.discountLabel'] = listing_data['priceInfo.discountLabel'].fillna(0)

    # drop the size column
    listing_data.drop('availableSizes', axis=1, inplace=True)

    # Join the images with path and add in the dataframe

    # Store the directory path in a varaible
    cutout_img_dir = cutOutImages_data
    model_img_dir = modelImages_data

    # list the directories
    cutout_images = os.listdir(cutout_img_dir)
    model_images = os.listdir(model_img_dir)

    listing_data['cutOutimageNames'] = listing_data['images.cutOut'].apply(lambda x: extractImageName(x))
    listing_data['modelimageNames'] = listing_data['images.model'].apply(lambda x: extractImageName(x))

    # Extract only those data points for which we have images
    listing_data = listing_data[listing_data['cutOutimageNames'].isin(cutout_images)]
    listing_data = listing_data[listing_data['modelimageNames'].isin(model_images)]

    # Add entire paths to cutOut and modelImages
    listing_data['cutOutImages_path'] = cutout_img_dir + '/' + listing_data['cutOutimageNames']
    listing_data['modelImages_path'] = model_img_dir + '/' + listing_data['modelimageNames']

    # Drop the cutOutimageNames, cutOutimageNames
    listing_data.drop(['cutOutimageNames', 'cutOutimageNames'], axis=1, inplace=True)
    listing_data.to_csv('listing_data_with_path.csv', index=False)

    # Extract the features for a all images
    index_values = np.random.randint(low=0, high=listing_data.shape[0]-1, size=20000)
    modelImages = listing_data.iloc[index_values]['modelImages_path']

    # Create the model object and extract the features of top 10000 images (VGG 16)
    vgg_feature_extractor = FeatureExtractor(arch='VGG')

    # dictionary to store the features and index of the image
    image_features_vgg = {}
    weights_to_save = []
    index_to_save = []
    total_images = len(index_values)
    for i, (idx, img_path) in enumerate(zip(index_values, modelImages)):
        print("{} images left".format(total_images - (i + 1)))
        # Extract features and store in a dictionary
        img = image.load_img(img_path)
        feature = vgg_feature_extractor.extract_features(img)
        image_features_vgg[idx] = feature
        weights_to_save.append(feature)
        index_to_save.append(idx)

    weights_to_save_arr = np.asarray(weights_to_save)
    index_to_save_arr = np.asarray(index_to_save)
    np.save('vgg_trained_features.npy', weights_to_save_arr)
    np.save('vgg_trained_index.npy', index_to_save_arr)