
### Supporting code for Computer Vision Assignment 1
### See "Assignment 1.ipynb" for instructions

import math

import numpy as np
from skimage import io

def load(img_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.
    HINT: Converting all pixel values to a range between 0.0 and 1.0
    (i.e. divide by 255) will make your life easier later on!

    Inputs:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, n_channels).
    """
    
    out = None
    img_array = io.imread(img_path)
    divimage = [i/255 for i in img_array]
    out = np.array(divimage)
    
    return out

def print_stats(image):
    """ Prints the height, width and number of channels in an image.
        
    Inputs:
        image: numpy array of shape(image_height, image_width, n_channels).
        
    Returns: none
                
    """
    print(image.shape)
    
    return None

def crop(image, x1, y1, x2, y2):
    """Crop an image based on the specified bounds. Use array slicing.

    Inputs:
        image: numpy array of shape(image_height, image_width, 3).
        (x1, y1): the coordinator for the top-left point
        (x2, y2): the coordinator for the bottom-right point
        

    Returns:
        out: numpy array of shape(x2 - x1, y2 - y1, 3).
    """

    out = None
    out = image[y1:y2,x1:x2]

    return out
    
def resize(input_image, fx, fy):
    """Resize an image using the nearest neighbor method.
    Not allowed to call the matural function.
    i.e. for each output pixel, use the value of the nearest input pixel after scaling

    Inputs:
        input_image: RGB image stored as an array, with shape
            `(image_height, image_width, 3)`.
        fx (float): the resize scale on the original width.
        fy (float): the resize scale on the original height.

    Returns:
        np.ndarray: Resized image, with shape `(image_height * fy, image_width * fx, 3)`.
    """
    out = None
    count = 0
    for i in input_image.shape:
        count+=1
    if count == 3:
        w,h,c = input_image.shape
    if count == 2:
        w,h = input_image.shape
    
    newW = int(fx * w)
    newH = int(fy * h)
    
    if count == 3:
        out = np.zeros((newW, newH, c))
    if count == 2:
        out = np.zeros((newW, newH))
    
    for i in range(newW):
        for j in range(newH):
            x = int(i/newW*w)
            y = int(j/newH*h)
            out[i][j] = input_image[x][y]
    return out


def change_contrast(image, factor):
    """Change the value of every pixel by following

                        x_n = factor * (x_p - 0.5) + 0.5

    where x_n is the new value and x_p is the original value.
    Assumes pixel values between 0.0 and 1.0 
    If you are using values 0-255, divided by 255.

    Inputs:
        image: numpy array of shape(image_height, image_width, 3).
        factor (float): contrast adjustment

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None
    out = factor * (image - 0.5) + 0.5

    return out

def greyscale(input_image):
    """Convert a RGB image to greyscale. 
    A simple method is to take the average of R, G, B at each pixel.
    Or you can look up more sophisticated methods online.
    
    Inputs:
        input_image: RGB image stored as an array, with shape
            `(image_height, image_width, 3)`.

    Returns:
        np.ndarray: Greyscale image, with shape `(image_height, image_width)`.
    """
    out = None
    R = input_image[:,:,0]
    G = input_image[:,:,1]
    B = input_image[:,:,2]
    out = (R+G+B)/3

    return out
    
def binary(grey_img, th):
    """Convert a greyscale image to a binary mask with threshold.
  
                  x_n = 0, if x_p < th
                  x_n = 1, if x_p > th
    
    Inputs:
        input_image: Greyscale image stored as an array, with shape
            `(image_height, image_width)`.
        th (float): The threshold used for binarization, and the value range is 0 to 1
    Returns:
        np.ndarray: Binary mask, with shape `(image_height, image_width)`.
    """
    out = None
    out = grey_img < th
    return out


def conv2D(image, kernel):
    """ Convolution of a 2D image with a 2D kernel. 
    Convolution is applied to each pixel in the image.
    Assume values outside image bounds are 0.
    
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    out = None
    #flip the kernel on both axes
    kernel = np.flip(kernel)
    Hk, Wk = kernel.shape
    #if the kernel is square
    Hi, Wi = image.shape
    
    out = np.zeros((Hi,Wi))
    for i in range(Hi-Hk+1):
        for j in range(Wi-Wk+1):
            out[i+1][j+1] = np.sum(image[i:i+Hk,j:j+Wk]*kernel)
    

    return out

def test_conv2D():
    """ A simple test for your 2D convolution function.
        You can modify it as you like to debug your function.
    
    Returns:
        None
    """

    # Test code written by 
    # Simple convolution kernel.
    kernel = np.array(
    [
        [1,0,1],
        [0,0,0],
        [1,0,0]
    ])

    # Create a test image: a white square in the middle
    test_img = np.zeros((9, 9))
    test_img[3:6, 3:6] = 1

    # Run your conv_nested function on the test image
    test_output = conv2D(test_img, kernel)   
    # print(test_output)

    # Build the expected output
    expected_output = np.zeros((9, 9))
    expected_output[2:7, 2:7] = 1
    expected_output[5:, 5:] = 0
    expected_output[4, 2:5] = 2
    expected_output[2:5, 4] = 2
    expected_output[4, 4] = 3

    # Test if the output matches expected output
    assert np.max(test_output - expected_output) < 1e-10, "Your solution is not correct."


def conv(image, kernel):
    """Convolution of a RGB or grayscale image with a 2D kernel
    
    Args:
        image: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
    """
    out = None
    count = 0
    for i in image.shape:
        count+=1
    if count == 3:
        R = conv2D(image[:,:,0],kernel)
        G = conv2D(image[:,:,1],kernel)
        B = conv2D(image[:,:,2],kernel)
        out = np.stack([R,G,B], axis = 2)
    if count == 2:
        out = conv2D(image[:,:],kernel)

    return out

    
def gauss2D(size, sigma):

    """Function to mimic the 'fspecial' gaussian MATLAB function.
       You should not need to edit it.
       
    Args:
        size: filter height and width
        sigma: std deviation of Gaussian
        
    Returns:
        numpy array of shape (size, size) representing Gaussian filter
    """

    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


def corr(image, kernel):
    """Cross correlation of a RGB image with a 2D kernel
    
    Args:
        image: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
    """
    out = None
    kernel = np.flip(kernel)
    out = conv2D(image, kernel)
    return out

