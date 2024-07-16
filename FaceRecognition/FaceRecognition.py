"""
Face detection using haar feature-based cascade classifiers
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# Training Data
#there is no label 0 /label 0 is empty
subjects = [ "","Alfredo", "Elvis", "Keanu"]


#function to detect face using OpenCV
def detect_face(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load cascade classifiers:
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

    # Detect faces:
    faces = face_cascade.detectMultiScale(gray);
    
    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None
    
    #under the assumption that there will be only one face, extract the face area
    (x, y, w, h) = faces[0]
    
    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]


#function to read all persons' training images, detect face from each image
#and will return two lists of exactly same size, one list 
# of faces and another list of labels for each face
def prepare_training_data(data_folder_path):
    
    #------STEP-1--------
    #get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)
    
    #list to hold all subject faces
    faces = []
    #list to hold labels for all subjects
    labels = []
    
    #let's go through each directory and read images within it
    for dir_name in dirs:
            
        #extract label number of subject from dir_name
        label = int(dir_name)
        print('label', label)
        
        #build path of directory containin images for current subject subject
        #sample subject_dir_path = "training-data/1"
        subject_dir_path = data_folder_path + "/" + dir_name
        print('Dir: ', subject_dir_path)
        
        #get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)
        
        #go through each image name, read image, 
        #detect face and add face to list of faces
        for image_name in subject_images_names:
            
            #ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;
            
            #sample image path = training-data/1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            #read image
            image = cv2.imread(image_path)
            
            #detect face
            face, rect = detect_face(image)

            #display an image window to show the image
            draw_rectangle(image, rect)
            cv2.imshow("Training on image...", image)
            cv2.waitKey(100)
            
            #------STEP-4--------
            #we will ignore faces that are not detected
            if face is not None:
                #add face to list of faces
                faces.append(face)
                #add label for this face
                labels.append(label)
                print('Ok ', image_name)
            
        print('Images: ', subject_images_names)
        print(' ')
        
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
    return faces, labels


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    

#this function recognizes the person in image passed
#and draws a rectangle around detected face with name of the 
#subject
def predict(test_img):
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face(img)

    #predict the image using our face recognizer 
    label, confidence = face_recognizer.predict(face)
    #get name of respective label returned by face recognizer
    text = subjects[label]
    
    #draw a rectangle around face detected
    draw_rectangle(img, rect)
    #draw name of predicted person
    cv2.putText(img, text, (rect[0], rect[1]), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    
    return img

#let's first prepare our training data
#data will be in two lists of same size
#one list will contain all the faces
#and other list will contain respective labels for each face
print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")

#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))


#create our LBPH face recognizer 
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#or use EigenFaceRecognizer by replacing above line with 
#face_recognizer = cv2.face.EigenFaceRecognizer_create()

#or use FisherFaceRecognizer by replacing above line with 
#face_recognizer = cv2.face.FisherFaceRecognizer_create()


#train our face recognizer of our training faces
face_recognizer.train(faces, np.array(labels))


# Now that we have the prediction function well defined, next step is to actually call this function on our test images and display those test images to see if our face recognizer correctly recognized them. So let's do it. This is what we have been waiting for. 
print("Predicting images...")

#load test images
test_img1 = cv2.imread("test-data/test1.jpg")
test_img2 = cv2.imread("test-data/test2.jpg")
test_img3 = cv2.imread("test-data/test3.jpg")

#perform a prediction
predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)
predicted_img3 = predict(test_img3)
print("Prediction complete")

#display both images
cv2.imshow("Img 1", predicted_img1)
cv2.imshow("Img 2", predicted_img2)
cv2.imshow("Img 3", predicted_img3)

cv2.waitKey(0)
cv2.destroyAllWindows()






