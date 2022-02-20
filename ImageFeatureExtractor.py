import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess
from tensorflow.keras.applications.xception import Xception, preprocess_input as xception_preprocess
from tensorflow.keras.models import Model
image.LOAD_TRUNCATED_IMAGES = True

# Creating a class for feature extraction and finding the most similar images

'''
Comparing 3 different models

1. VGG 16
2. ResNet 50
3. Xception
'''


class FeatureExtractor:

    # Constructor
    def __init__(self, arch='VGG'):

        self.arch = arch

        # Using VGG -16 as the architecture with ImageNet weights
        if self.arch == 'VGG':
            base_model = VGG16(weights='imagenet')
            self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

        # Using the ResNet 50 as the architecture with ImageNet weights
        elif self.arch == 'ResNet':
            base_model = ResNet50(weights='imagenet')
            self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

        # Using the Xception as the architecture with ImageNet weights
        elif self.arch == 'Xception':
            base_model = Xception(weights='imagenet')
            self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

    # Method to extract image features
    def extract_features(self, img):

        # The VGG 16 & ResNet 50 model has images of 224,244 as input while the Xception has 299, 299
        if self.arch == 'VGG' or self.arch == 'ResNet':
            img = img.resize((224, 224))
        elif self.arch == 'Xception':
            img = img.resize((299, 299))

        # Convert the image channels from to RGB
        img = img.convert('RGB')

        # Convert into array
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        if self.arch == 'VGG':
            # Proprocess the input as per vgg 16
            x = vgg_preprocess(x)

        elif self.arch == 'ResNet':
            # Proprocess the input as per ResNet 50
            x = resnet_preprocess(x)

        elif self.arch == 'Xception':
            # Proprocess the input as per ResNet 50
            x = xception_preprocess(x)

        # Extract the features
        features = self.model.predict(x)

        # Scale the features
        features = features / np.linalg.norm(features)

        return features
