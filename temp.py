# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
from moviepy.editor import VideoFileClip
import os
#%matplotlib inline



def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    top=320
    bottom=540
    left_x1_vec = []
    left_y1_vec = []
    left_x2_vec = []
    left_y2_vec = []
    right_x1_vec = []
    right_y1_vec = []
    right_x2_vec = []
    right_y2_vec = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            #cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            slope = ((y2-y1)/(x2-x1))
            # Ignore obviously invalid lines
            if slope > -0.8 and slope < -0.5:
                left_x1_vec.append(x1)
                left_y1_vec.append(y1)
                left_x2_vec.append(x2)
                left_y2_vec.append(y2)
            if slope > 0.5 and slope < 0.8:
                right_x1_vec.append(x1)
                right_y1_vec.append(y1)
                right_x2_vec.append(x2)
                right_y2_vec.append(y2)
                
    avg_right_x1 = int(np.mean(right_x1_vec))
    avg_right_y1 = int(np.mean(right_y1_vec))
    avg_right_x2 = int(np.mean(right_x2_vec))
    avg_right_y2 = int(np.mean(right_y2_vec))
    right_slope = ((avg_right_y2-avg_right_y1)/(avg_right_x2-avg_right_x1))
    #get all line parameters  y-y0=m(x-x0)
    right_y1 = top
    right_x1 = int(avg_right_x1 + (right_y1 - avg_right_y1) / right_slope)
    right_y2 = bottom
    right_x2 = int(avg_right_x1 + (right_y2 - avg_right_y1) / right_slope)
    cv2.line(img, (right_x1, right_y1), (right_x2, right_y2), color, thickness)
        
    avg_left_x1 = int(np.mean(left_x1_vec))
    avg_left_y1 = int(np.mean(left_y1_vec))
    avg_left_x2 = int(np.mean(left_x2_vec))
    avg_left_y2 = int(np.mean(left_y2_vec))
    left_slope = ((avg_left_y2-avg_left_y1)/(avg_left_x2-avg_left_x1))
    #get all line parameters  y-y0=m(x-x0)
    left_y1 = top
    left_x1 = int(avg_left_x1 + (left_y1 - avg_left_y1) / left_slope)
    left_y2 = bottom
    left_x2 = int(avg_left_x1 + (left_y2 - avg_left_y1) / left_slope)
    cv2.line(img, (left_x1, left_y1), (left_x2, left_y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def process_img(img):
    gray = grayscale(img)
    blurred = gaussian_blur(gray, kernel_size=5)
    edges = canny(blurred, low_threshold=50, high_threshold=150)
    
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(450, 320), (500, 320), (imshape[1],imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    lines = hough_lines(masked_edges, rho=2, theta=np.pi/180, threshold=15, min_line_len=40, max_line_gap=20)

    lines_edges = weighted_img(lines, img) 
    return lines_edges
    #weighted_img(lines, initial_img, α=0.8, β=1., γ=0.)
    
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
#clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
#white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!

if __name__ == "__main__":
    #reading in an image
    image = mpimg.imread('test_images/solidWhiteRight.jpg')
    #printing out some stats and plotting
    print('This image is:', type(image), 'with dimensions:', image.shape)
    plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
    lines_edges = process_img(image)
    plt.imshow(lines_edges)
    
    images = os.listdir("test_images/")
    for img_file in images:
        #print(img_file)
        # Skip all files starting with line.
        if img_file[0:4] == 'line':
            continue

        img = mpimg.imread('test_images/' + img_file)   

        weighted = process_img(img)

        plt.imshow(weighted)
        #break
        mpimg.imsave('test_images/lines-' + img_file, weighted)

    #white_output = 'test_videos_output/solidWhiteRight.mp4'

