import numpy as np
import cv2 as cv
import time

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((4*4,3), np.float32)
objp[:,:2] = np.mgrid[0:4,0:4].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

vc= cv.VideoCapture()
vc.open(0)

detections=0
while detections<30:
    _,img = vc.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (4,4), None)
    cv.imshow('img', img)
    # If found, add object points, image points (after refining them)
    if ret == True:
        print("found checkerboard")
        objpoints.append(objp)
        detections= detections+1
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (4,4), corners2, ret)
        cv.waitKey(500)
    time.sleep(3)
cv.destroyAllWindows()
vc.release()

print("done with that shite")
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("cam matrix: \n",mtx)
print("distortionCoeffs: \n",dist)