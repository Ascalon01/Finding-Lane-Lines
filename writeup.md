# **Finding Lane Lines on the Road** 

## Reflection

## 1. Pipeline description
My pipeline consists of 7 steps:</br>
1. [Reading image or video frame](#reading-image-or-video-frame)</br>
2. [Filtering white and yellow colors](#filtering-white-and-yellow-colors)</br>
3. [Gaussian blurring](#gaussian-blurring)</br>
4. [Edge detection](#edge-detection)</br>
5. [Region of interest definition](#region-of-interest-definition)</br>
6. [Hough lines detection](#gaussian-blurring)</br>
7. [Finding Lane Lines](#finding-lane-lines)</br>

</br>

### Reading image or video frame
We have "flagvideo" that allows us to choose between if its a videos or images to be processed,

if the flagvideo is set true , then it reads all the videos in "test_videos/" Directory with below code.

```python
files=glob.glob(os.path.join('test_videos/','*.mp4'))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    for i in files:
        tmpname=str.split(i,'\\')
        tmpname=str.split(tmpname[1],'.')
        cap = cv2.VideoCapture(i)
```
if the flagvideo is set to false , then it reads all images in the "test_images/" Directory with below code.

```python
files=glob.glob(os.path.join('test_images/','*.jpg'))
    for i in files:
        tmpname=str.split(i,'\\')
        tmpname=str.split(tmpname[1],'.')
        img=cv2.imread(i)
```
---
### Filtering white and yellow colors
This step wouldn't be necessary for the first two videos probably gray scale conversion should be sufficient enough. However,In the Challenge video, the gray scale conversion wouldn't work . We would like to differentiate white and yellow , Thus the idea of initial filtering of 2 key colors which are the main components of the road lanes.
Firstly, We would like to pick the yellow color in rgb color space and convert the image into HSL color space to pick white color.
You can probably find the color in RGB color space using pick colors from paint.
The same is performed with below code.

```python
def hsv(img):
    
    img1 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    yellow1 = np.array([ 0,120, 120], dtype=np.uint8)
    yellow2 = np.array([ 100,255, 255,], dtype=np.uint8)
    yellow = cv2.inRange(img, yellow1, yellow2)
    white1 = np.array([0, 0, 200], dtype=np.uint8)
    white2 = np.array([255, 30, 255], dtype=np.uint8)
    white = cv2.inRange(img1, white1, white2)
    out=cv2.bitwise_and(img, img, mask=(yellow | white))
    return out
```
```python
hsvimg=hsv(img)
```
---
### Gaussian blurring
To supress noise and spurious gradients Gaussian smoothing is applied.kernel of size 3 was chosen. Here, it's again preparation for edge detection step. Borders between lane and road can be not so smooth, so we don't want the edge detector to classify such regions as additional lines.
```python
def gaussian_blur(img, kernel_size):
   
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
```    
```python
blur=gaussian_blur(hsvimg,3)
```
---
### Edge detection
To detect edges, let's use popular [Canny](http://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=canny#canny) method. It's called with 2 parameters: low and high thresholds which should be found by trial and error. According to the OpenCV documentation:
<ul>
	<li>If a pixel gradient is higher than the upper threshold, the pixel is accepted as an edge</li>
	<li>If a pixel gradient value is below the lower threshold, then it is rejected.</li>
	<li>If the pixel gradient is between the two thresholds, then it will be accepted only if it is connected to a pixel that is above the upper threshold.</li>
</ul>

Canny recommended a upper:lower ratio between 2:1 and 3:1. I chose values of 80 and 60. Below, there are outputs of this operation.

```python
def canny(img, low_threshold, high_threshold):
    
    return cv2.Canny(img, low_threshold, high_threshold)

```
```python
edge=canny(blur,60,80)
```
---
### Region of interest definition
To filter out unnecessary objects in the image, the region of interest is defined. Such mask (here it's trapezoid) is then applied to the working image.
```python
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)     
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
```
if flagvideo is set to true,
```python
cap = cv2.VideoCapture(i)
width = int(cap.get(3) )
height = int(cap.get(4))
midx=width/2
maxt=height/2+30
vertices = np.array([[(30,height),(midx-30, maxt), (midx+30, maxt), (width-30,height)]], dtype=np.int32)
```
if flagvideo is set to false,
```python
img=cv2.imread(i)
width = img.shape[1]
height = img.shape[0]
midx=width/2
maxt=height/2+30
vertices = np.array([[(30,height),(midx-30, maxt), (midx+30, maxt), (width-30,height)]], dtype=np.int32)
```

---
### Hough lines detection

Now, having edges detected in our interest area, all straight lines need to be identified. This is done by . This operation has quite many parameters which need to be tuned experimentally.  

Now, having edges detected in our interest area, all straight lines need to be identified. This is done by [Hough transform](http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/hough_lines/hough_lines.html). This operation has quite many parameters which need to be tuned experimentally. Speaking at high level, they define how long or how "straight" the sequence of pixels should be to be classified as one line.

```python
rho=2
theta=np.pi/180
threshold=50
min_line_len=15
max_line_gap=10

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    find_lanes(line_img, lines)
    return line_img
```
```python
himg=hough_lines(roiimg, rho, theta, threshold, min_line_len, max_line_gap)
```

---
### Finding lane lines
Small horizontal/Vertical lines appearing inside the region of interest (each Hough line) we calculate a slope parameter.  Based on the slope parameter we differentiate if the hough line belongs to left lane or right lane.
When the difference between average slope of lane and slope calculated is lower than 0.1 we append those points to a list which consists of right points and left points.

```python
difference = 0.1
    leftSlopeAvg = 0
    pointsLeft = []
    l = 1

    rightSlopeAvg = 0
    pointsRight = []
    r = 1
    
    yy=[]

    for line in lines:
        for x1,y1,x2,y2 in line:
            dx=x2-x1
            if dx!=0:
                slope = (y2-y1)/dx
                if slope < 0:
                    rightSlopeAvg = rightSlopeAvg + (slope - rightSlopeAvg) / r
                    if np.absolute(rightSlopeAvg - slope) < difference:
                        pointsRight.append((x1, y1))
                        pointsRight.append((x2, y2))
                        yy.append(y1)
                        yy.append(y2)
                    r += 1
                else:
                    leftSlopeAvg = leftSlopeAvg + (slope - leftSlopeAvg) / l
                    if np.absolute(leftSlopeAvg - slope) < difference:
                        pointsLeft.append((x1, y1))
                        pointsLeft.append((x2, y2))
                        yy.append(y1)
                        yy.append(y2)
                    l += 1
```
we try to fit line for all the right and left points gathered and find the intercept as well as slope and also add these calculated data to a deque variable previous_frames inorder to avoid jitters by taking mean .
once the intercept and slope is found, we apply the line equation to find the x-cordinate.
we know that the y co-ordinate can be found from the list of points appended to yy and size of image/frame .
by applying Y=mX+C, we find the x-cordinates and try to draw lines for the same.
if incase we fail to find the lines that match our condition we try drawing lines directly from hough transform.

```python
if len(pointsRight) > 0 and len(pointsLeft) > 0:
  # right lane
  [vx, vy, x, y] = cv2.fitLine(np.array(pointsRight, dtype=np.int32), cv2.DIST_L2, 0, 0.01, 0.01)
  rightSlope = vy / vx
  rightIntercept = y - (rightSlope * x)

  # left lane
  [vx, vy, x, y] = cv2.fitLine(np.array(pointsLeft, dtype=np.int32), cv2.DIST_L2, 0, 0.01, 0.01)
  leftSlope = vy / vx
  leftIntercept = y - (leftSlope * x)

  previous_frames.append((rightIntercept, rightSlope, leftIntercept, leftSlope))

try:
  if len(previous_frames) > 0:
      avg = np.sum(previous_frames, -3) / len(previous_frames)
      rightIntercept = avg[0]
      rightSlope = avg[1]
      leftIntercept = avg[2]
      leftSlope = avg[3]

  startY = max(yy)
  endY = int(img.shape[0]/1.6) 

  rightX1 = (startY - rightIntercept) / rightSlope
  rightX2 = (endY - rightIntercept) / rightSlope   
  leftX1 = (startY - leftIntercept) / leftSlope
  leftX2 = (endY - leftIntercept) / leftSlope

  cv2.line(img, (rightX1, startY), (rightX2, endY), color, thickness)
  cv2.line(img, (leftX1, startY), (leftX2, endY), color, thickness)

except Exception  as e:

  for line in lines:
      for x1,y1,x2,y2 in line:
          cv2.line(img, (x1, y1), (x2, y2), color, thickness)
return img
```
---
    
## 2. Potential shortcomings
Potential shortcoming of this algorithm is , 
1. We have strictly defined the region of interest and lane colors, The algorithm would not work so well on roads with sign marked in yellow and white.
2. The algorithm might not work well for lanes that are curvy.

## 3. Possible improvements

Possible improvement would be,
1. To use some kind of higher order polynomial fit to handle curvy lanes. 
2. instead of just defining many parameters used here, maybe we can automate the process of parameters search. 
