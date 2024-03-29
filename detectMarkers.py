import cv2
import numpy as np
import math
import time
from pynput.mouse import Controller
from scipy import signal

showVid= False
vid = cv2.VideoCapture()
vid.open(0)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
distortionCoeffs = np.float32([ 0.0117417,  -0.12165514,  0.0033618,  -0.00090678, 0.19157184])
cameraMatrix = np.float32( [[497.28758068, 0, 332.32358251] ,[0, 496.66656206, 265.90375616],[ 0, 0, 1 ]] )

def testArUcoDetection():
    ref,img = vid.read()
    corners,ids,_ = cv2.aruco.detectMarkers(img,dictionary)

    if(ids is not None):
        rvecs,tvecs,_ = cv2.aruco.estimatePoseSingleMarkers(corners,12.75,cameraMatrix,distortionCoeffs)
        if (showVid): 
            img = cv2.aruco.drawAxis(img,cameraMatrix,distortionCoeffs,rvecs[0],tvecs[0],10)
    
    if (cv2.waitKey(5)>0 or len(img)==0):
        return
    cv2.imshow("WINNAME",img)

def getMarkerTransform(img):
    corners,ids,_ = cv2.aruco.detectMarkers(img,dictionary)

    if(ids is not None):
        rvecs,tvecs,_ = cv2.aruco.estimatePoseSingleMarkers(corners,12.75,cameraMatrix,distortionCoeffs)
        dst,_ = cv2.Rodrigues(rvecs[0])
        fwdVector= np.matmul( dst,np.array([0,0,1]) )
        pointer = np.matmul( dst,np.array([0,9,0]) ) + tvecs[0]     #as sights are on top of marker
        return pointer, fwdVector
    else:
        return None,None

def getPointOfIntersectionLines():
    P1= None
    P2= None
    while(P1 is None):
        print("P1 not set, presss Enter when you want to calibrate") 
        input()
        img=[]
        while(img.length==0):
            _,img = vid.read()
        P1,V1 = getMarkerTransform(img)
    while(P2 is None):
        print("P2 not set, presss Enter when you want to calibrate") 
        input()
        img=[]
        while(img.length==0):
            _,img = vid.read()
        P2,V2 = getMarkerTransform(img)
    V3 =  np.cross(V1,V2)
    VMat = np.array( [ [V1[0],V1[1],V1[2]] , [V2[0],V2[1],V2[2]] , [V3[0],V3[1],V3[2]]  ])
    tVec = np.matmul( P2-P1 ,np.linalg.inv(VMat))
    
    POI = P1+ tVec[0,0]*V1 + tVec[0,2]*V3*0.5
    return POI[0]
    
def calibrateScreen(shots):
    print("Fire Twice at bottom left")
    bottomLeft = getPointOfIntersectionLines()
    print("Fire Twice at bottom right")
    bottomRight = getPointOfIntersectionLines()
    print("Fire Twice at top left")
    TopLeft = getPointOfIntersectionLines()
    return bottomLeft,bottomRight,TopLeft

def getNoramlizedScreenpos(BL,BR,TL,P,Fwd):
    planeNormal = np.cross(BR-BL,TL-BL) # screen x axis cross screen y axis
    planeNormal = planeNormal/ np.linalg.norm(planeNormal)
    d = np.dot(BL- P,planeNormal)/np.dot(Fwd,planeNormal)
    POI = P+d*Fwd
    FromBL = POI-BL
    ScreenXaxis = (BR-BL)/np.linalg.norm(BR-BL)
    ScreenYaxis = (TL-BL)/np.linalg.norm(TL-BL)
    x = np.dot(FromBL,ScreenXaxis)/np.linalg.norm(BR-BL)
    y = np.dot(FromBL,ScreenYaxis)/np.linalg.norm(TL-BL)
    return x,y

#BL,BR,TL,= calibrateScreen()
BL= np.array([18,21,0])
BR= np.array([-18,21,0])
TL= np.array([18,0,0])
print("Screen points positions: ",BL,BR,TL)

mouse = Controller()
print("take mouse to bottom right and press enter")
input()
xMax,yMax = mouse.position
print(xMax,yMax)

xArray= np.zeros(10)
yArray= np.zeros(10)
sos = signal.butter(10,0.35,output='sos')

LastFrameTime=2
while True:
    #testArUcoDetection()
    _,img = vid.read()
    if (img is None):
        print("img is none")
        break
    if (img.size==0):
        print("empty frame detected")
        break

    getTransformStartTime = time.process_time()
    P,Fwd =  getMarkerTransform(img)
    print ( "getMarkerTransform time: "+str(time.process_time()-getTransformStartTime))
    
    
    if (P is not None):
        getScreenposTime = time.process_time()
        x,y = getNoramlizedScreenpos(BL,BR,TL,P,Fwd)
        print ( "getScreenpos time: "+str(time.process_time()-getScreenposTime))
    else:
        x= xArray[9]
        y= yArray[9]

    xArray =  np.roll(xArray,1)    #just right shifting the array and adding latest value of x
    xArray[9] = x
    yArray =  np.roll(yArray,1)
    yArray[9] = y
    xFiltered = signal.sosfilt(sos,xArray)[9]
    yFiltered = signal.sosfilt(sos,yArray)[9]
    
    mouse.position = (xFiltered*xMax, (1-yFiltered) *yMax)
    
    print("FPS: ", 1/(time.process_time()-LastFrameTime+0.00001), " Hz" )   # the 0.0001 is added to remove division by zero errors
    LastFrameTime = time.process_time()

    
cv2.destroyAllWindows()
vid.release()