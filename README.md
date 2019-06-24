# **Finding Lane Lines on the Road** 

### **Project Overview**

The goal of the project is to create an image processing algorithm using basic computer vision techniques, including Canny edge detection and Hough transforms, to correctly identify lane lines on the road. The algorithm should be robust enough to work on a series of images (video) with varying conditions like changing light conditions and color variations from shadows, etc.


#### Files in the repository
* Code for the algorithm is contained in the [IPython notebook](./Finding%20Lane%20Lines.ipynb) as well as python script "Finding Lane Lines.py"

* A [writeup](./writeup.md) detailing the results of the project and describing the procedure for deriving a single straight line corresponding to a series of line segments found by the Hough transform.

* The processed videos (the same algorithm was used to process each video):
  * A video with a [solid white lane on the right](./output_videos/Annotated_solidWhiteRight.mp4)
  * A video with a [solid yellow lane on the left](./output_videos/Annotated_solidYellowLeft.mp4)
  * A video with [more challenging features](./output_videos/Annotated_challenge.mp4), such as shadows occluding the lane lines and cars in the adjacent lanes

#### Running the code
This project was developed using Python 3.5. The IPython notebook can be run using [Jupyter Notebooks](http://jupyter.org/).
The project depends on the,
  1. [NumPY](http://www.numpy.org/)
  2. [OpenCV](http://opencv.org/)
