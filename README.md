# Self-Driving Car Engineer Nanodegree


## Project: **Finding Lane Lines on the Road** 

### Packages needed
1. Matplotlib
2. Numpy
3. cv2
4. math
5. moviepy for visualizng the video

### Steps for detecting Lanes in a image
1. Convert the given image to gray scale.
2. Blur the grayscale image to make the patterns in the image even.
3. we need to fing the edges in the images, so we can use canny algorithm which uses change in gradient to find the edges in a image.
4. We eliminate un wanted lines by takeing only the region of interest in the image.
5. then we find the lines in the region of interest with the edges detected by hought lines.
6. Finally we put the detected line on top of the original image.

### Problems with the current pipeline
1. horizontal lines may cause a problem.
2. colour of the lane may also be a problem.
3. we are not sure if the hyper parameters which we used will help in all kind of lanes. 

### Possible solutions
1. Try to hardcode different colours for lanes(as this is not a AI task)
2. take care of horizontal lines by making a condition of y2-y1 < some constant.
