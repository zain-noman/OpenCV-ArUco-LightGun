# OpenCV-ArUco-LightGun
Control your cursor so it points wherever you are pointing your gun!
The algorithm works by first detecting the position and orintation of the aruco marker. Then it calculates where the forward vector of the aruco gun intersects with the screen which is treated as a plane. An IIR filter is then further added to smooth the cursor motion.
This implementation has the advantage that it responds to tilt changes as well as position changes as compared to other methods that can not be used for lightguns.
A camera calibration script is also provided. The camera calibration script can be used to set the cameraMatrix and DistortionCoeffs. 
