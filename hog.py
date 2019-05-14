# -*- coding: utf-8 -*-
import cv2
from skimage.feature import hog
from PIL import Image

scale = 10
def generate_hog(filename):
    im = cv2.imread(filename)
    gr = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
    image = 255-gr
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=True)
    return hog_image

def save_hog(filename, hog_image):
    name=filename[:-4]+"__HOG.png"
    img = Image.fromarray(hog_image*scale).convert('L')
    img.save(name)
    
img_hog = generate_hog("images/lambert rosique.jpg")
save_hog("hog.png",img_hog)