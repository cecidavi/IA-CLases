import cv2
import numpy as np
from matplotlib import pyplot as plt
import face_recognition

img1 = cv2.imread("test1.jpg")
img2 = cv2.imread("test2.jpg")

encodings1 = face_recognition.face_encodings(img1)
encodings2 = face_recognition.face_encodings(img2)
# Create data (three different 'clusters' of points (it should be of np.float32 data type):
data = np.float32(np.vstack((
    encodings1,
    encodings2)))
print(len(data))
print(len(data[0]))
print(len(data[1]))
print("")
#data[99,:] = [24, 24]
plt.plot(data[0], c = 'b')
plt.plot(data[1], c = 'r')
plt.show()

# K means
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
ret, label, center = cv2.kmeans(data, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

print(label)

# Now separate the data using label output
A = data[label.ravel() == 0]
B = data[label.ravel() == 1]
#C = data[label.ravel() == 2]
print(len(A))
print(len(B))
# plot it
plt.plot(A, c='b')
plt.plot(B, c='g')
#plt.scatter(C[:, 0], C[:, 1], c='c')
#plt.scatter(center[:, 0], center[:, 1], s=100, c='r')
plt.show()
