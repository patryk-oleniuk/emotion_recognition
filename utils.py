import sys
#workaround for openCV on osX
sys.path.append('/usr/local/lib/python3.6/site-packages') 

# IMPORTANT: OPENCV 3 for Python 3 is needed, install it from : 
# http://docs.opencv.org/3.0-beta/doc/tutorials/introduction/windows_install/windows_install.html
# or on MAC : brew install opencv3 --with-contrib --with-python3 --HEAD
# http://www.pyimagesearch.com/2016/12/19/install-opencv-3-on-macos-with-homebrew-the-easy-way/

import cv2
import matplotlib.pyplot as plt
import PIL 
import numpy as np

from scipy.sparse import coo_matrix

# returns the array of 48x48 images of faces and the whole image with rectangles over the faces
def get_faces_from_img(img_path):
    
    # The face recognition properties, recognizing only frontal face
    cascPath = 'haarcascade_frontalface_default.xml'
    
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)

    #read image and convert to grayscale
    if (img_path == 'camera'):
        video_capture = cv2.VideoCapture(0)
        ret, image = video_capture.read()
    else:
        image = cv2.imread(img_path,1)
    
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    #print("Found {0} faces in ".format(len(faces)), img_path, " !")

    #preparing an array to store each face img separately 
    faces_imgs = np.zeros((len(faces),48,48))

    # iterate through the faces and save them into a separate array
    num_fac = 0;

    for (x, y, w, h) in faces:

        face_single = image[y:y+h,x:x+w];
        #resize to 48x48
        face_resized = cv2.resize(face_single, (48,48));
        #cv2.imwrite('Face'+str(num_fac)+'.png', face_resized)
        #taking only one color (because it's grey RGB)
        faces_imgs[num_fac] = face_resized[:,:,0]
        num_fac = num_fac+1;
        #adding rectangles to faces

    # adding rectangles on faces in the image
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)
    #cv2.imshow("Faces found", image)
    #cv2.imwrite('Faces_recognized.png', image)
    return faces_imgs, image

def convert_to_one_hot(a,max_val=None):
    N = a.size
    data = np.ones(N,dtype=int)
    sparse_out = coo_matrix((data,(np.arange(N),a.ravel())), shape=(N,max_val))
    return np.array(sparse_out.todense())

#convert single vector to a value
def convert_from_one_hot(a):
     return np.argmax(a);
    
def get_min_max( img ):
    return np.min(np.min(img)),np.max(np.max(img))

def remap(img, min, max):
    if ( max-min ):
        return (img-min) * 1.0/(max-min)
    else:
        return img

def contrast_stretch(img ):
    min, max = get_min_max( img );
    return remap(img, min, max)