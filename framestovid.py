import cv2
import numpy as np
import glob
from tqdm import tqdm

img_array = []
for filename in glob.glob('I:\\ML\\Projs\\videoclass\\stitch\\*.jpg'):
    img = cv2.imread(filename)
    img_array.append(img)

height, width, layers = img.shape
size = (width,height)
 
 
out = cv2.VideoWriter('outVideo.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in tqdm(range(len(img_array))):
    out.write(img_array[i])
out.release()