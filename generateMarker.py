import cv2
import numpy as np

img =np.zeros([600,600],float)
dictionary= cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
img = cv2.aruco.drawMarker(dictionary, 23, 300, 1)
print(img.shape)
cv2.imshow("yo",img)
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imwrite("marker.png",img)