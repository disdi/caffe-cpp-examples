import sys
import random
import os
import lmdb
import cv2
import numpy as np

directory = os.getcwd()
datadir = '\\images\\'
dataset = '\\images\\train.txt'

poses = []
images = []

with open(directory+dataset) as f:
    for line in f:
        fname = line.split()
        images.append(directory+datadir+fname[0])

print(images[0])		
print(images[1])		
print(images[2])		
r = list(range(len(images)))
random.shuffle(r)

print 'Creating Dataset.'
env = lmdb.open('dataset_lmdb', map_size=int(12))

count = 0

for i in r:
    print 'Saving image: ', count+1
    X = cv2.imread(images[i])
    res = cv2.resize(X,(0,0), fx=0.5, fy=0.5)
    count = count+1

env.close()