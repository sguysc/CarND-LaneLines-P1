# **Finding Lane Lines on the Road** 

## Writeup

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road

* Reflect on your work in a written report

  

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. 

1. First, I converted the images to grayscale, then I used Gaussian blur  (with kernel_size=5) to prepare the image for edge detect.
2. I used canny edge detect on the blurred image with the following thresholds :low_threshold=50, high_threshold=150.
3. I set the ROI to roughly the lower half of the image (using the region_of_interest function)
4. I used the hough transform (hough_lines function) for straight line extraction (with parameters: rho=2, theta=np.pi/180, threshold=15, min_line_len=40, max_line_gap=2).
5. plotted the result from 4 using weighted_img (on the original image).

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by:

1. I split the detected lines into two groups (left and right lines):
   a. lines with slope $\left(-0.8,-0.5\right)$
   b. lines with slope $\left(0.5,0.8\right)$

2. I averaged over all the slopes of each line for each group:

   a. left_slope

   b. right_slope

3. I added the lines to the input image in such a way that each line passes through the point which is the average of all points in the group.   



### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when crossing over a lane, then there would be a line with  a slope which does not fall into any of the slope intervals noted above.

Another shortcoming could be if one of the lines would intersect with the right edge of the image then I would try to draw a segment which its lower endpoint is outside the image.




### 3. Suggest possible improvements to your pipeline

A possible improvement would be to check for outliers. Obtain a line estimation as done above and assess which of the lines returned from hough is too far from the estimated, remove it and re-estimate.

Another potential improvement could be to implement a polynomial fit to the lines returned from the hough transform.

Another potential improvement would be to use Deep Learning to find lanes, but that would be getting ahead of myself... :-)

