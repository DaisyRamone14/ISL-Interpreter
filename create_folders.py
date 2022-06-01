import os
import string

if not os.path.exists("dataset"):
    os.makedirs("dataset")

if not os.path.exists("dataset/training"):
    os.makedirs("dataset/training")

if not os.path.exists("dataset/testing"):
    os.makedirs("dataset/testing")

for i in range(0):
    if not os.path.exists("dataset/training/" + str(i)):
        os.makedirs("dataset/training/" + str(i))

    if not os.path.exists("dataset/testing/" + str(i)):
        os.makedirs("dataset/testing/" + str(i))

#A to Z folders in the training and testing folders 

for i in string.ascii_uppercase:
    if not os.path.exists("dataset/training/" + i):
        os.makedirs("dataset/training/" + i)
    
    if not os.path.exists("dataset/testing/" + i):
        os.makedirs("dataset/testing/" + i)