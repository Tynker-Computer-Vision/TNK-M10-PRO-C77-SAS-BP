import os
import pickle
import numpy as np

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow import keras

import cv2
import numpy as np

features = 'null'

# load features from features.pkl


# Load the captions from caption.txt


# Load the model


# Create mapping of image to captions

# Loop through every caption

    # Split the line by comma(,)
    
    # Move to next iteration if length of line is less then 2 characters
    
    # Take image_id and caption from token[0], [1] respectively
    
    # Remove extension from image ID
    
    # Convert caption list to string
    
    # Create list if needed
    
    # Store the caption
    

# Print the mapping dictionary




