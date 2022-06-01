import cv2
import numpy as np
import os
import string
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

mode = 'testing'
directory = 'dataset/'+mode+'/'

sdir = 'aug_dataset/'+mode+'/'

if not os.path.exists("aug_dataset"):
    os.makedirs("aug_dataset")

if not os.path.exists("aug_dataset/training"):
    os.makedirs("aug_dataset/training")

if not os.path.exists("aug_dataset/testing"):
    os.makedirs("aug_dataset/testing")

#A to Z folders in the training and testing folders 

for i in string.ascii_uppercase:
    if not os.path.exists("aug_dataset/training/" + i):
        os.makedirs("aug_dataset/training/" + i)
    
    if not os.path.exists("aug_dataset/testing/" + i):
        os.makedirs("aug_dataset/testing/" + i)
        

#Count of existing images
count = {       
    'a': len(os.listdir(directory+"/A")),
    'b': len(os.listdir(directory+"/B")),
    'c': len(os.listdir(directory+"/C")),
    'd': len(os.listdir(directory+"/D")),
    'e': len(os.listdir(directory+"/E")),
    'f': len(os.listdir(directory+"/F")),
    'g': len(os.listdir(directory+"/G")),
    'h': len(os.listdir(directory+"/H")),
    'i': len(os.listdir(directory+"/I")),
    'j': len(os.listdir(directory+"/J")),
    'k': len(os.listdir(directory+"/K")),
    'l': len(os.listdir(directory+"/L")),
    'm': len(os.listdir(directory+"/M")),
    'n': len(os.listdir(directory+"/N")),
    'o': len(os.listdir(directory+"/O")),
    'p': len(os.listdir(directory+"/P")),
    'q': len(os.listdir(directory+"/Q")),
    'r': len(os.listdir(directory+"/R")),
    's': len(os.listdir(directory+"/S")),
    't': len(os.listdir(directory+"/T")),
    'u': len(os.listdir(directory+"/U")),
    'v': len(os.listdir(directory+"/V")),
    'w': len(os.listdir(directory+"/W")),
    'x': len(os.listdir(directory+"/X")),
    'y': len(os.listdir(directory+"/Y")),
    'z': len(os.listdir(directory+"/Z"))
}

# Initialising the ImageDataGenerator class.
datagen = ImageDataGenerator(
        rotation_range = 40,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = False,
        brightness_range = (0.5, 1.5))

#choose from A to Z
for i in range(65,91):     
    #traversing through all the images in a alphabet
    for j in range(1,count[chr(i+32)]): 
        img = tf.keras.preprocessing.image.load_img(directory+"/"+str(chr(i))+"/"+str(j)+".jpg")
        # Converting the input sample image to an array
        x = tf.keras.preprocessing.image.img_to_array(img)
        # Reshaping the input image
        x = x.reshape((1, ) + x.shape) 
        # Generating and saving 5 augmented samples//using the above defined parameters. 
        k = 0
        for batch in datagen.flow(x, batch_size = 1,
                                save_to_dir = sdir+"/"+chr(i), 
                                save_prefix =len(os.listdir(sdir+str(chr(i))))+1, save_format ='jpeg'):
            k += 1
            if k > 5:
                break