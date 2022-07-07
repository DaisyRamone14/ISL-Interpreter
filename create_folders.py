import os
import string

if not os.path.exists("dataset"):
    os.makedirs("dataset")

#A to Z folders

for i in string.ascii_uppercase:
    if not os.path.exists("dataset/" + i):
        os.makedirs("dataset/" + i)
    