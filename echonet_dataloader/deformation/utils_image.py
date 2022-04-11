from __future__ import print_function
import math
import os
import random
import copy
import scipy
import imageio
import string
import numpy as np
from skimage.transform import resize
try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * (t ** (n - i)) * (1 - t) ** i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])
    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)
    return xvals, yvals


def data_augmentation(x, y, prob=0.5):
    # augmentation by flipping
    cnt = 1
    while random.random() < prob and cnt > 0:
        degree = random.choice([0, 1])
        # print('augmentation')
        x = np.flip(x, axis=degree)
        y = np.flip(y, axis=degree)
        cnt = cnt - 1
    return x, y


def nonlinear_transformation(x, prob=0.5):
    if random.random() >= prob:
        return x
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=100)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    # print('nonlinear_transformation')
    return nonlinear_x


def local_pixel_shuffling(x, prob=0.5):
    # print(x)
    if random.random() >= prob:
        return x
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    img_deps, img_rows, img_cols = x.shape
    num_block = 800
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows // 20)
        block_noise_size_y = random.randint(1, img_cols // 20)
        # block_noise_size_z = random.randint(1, img_deps//10)
        noise_x = random.randint(0, img_rows - block_noise_size_x)
        noise_y = random.randint(0, img_cols - block_noise_size_y)
        # noise_z = random.randint(0, img_deps-block_noise_size_z)
        window = orig_image[0, noise_x:noise_x + block_noise_size_x,
                 noise_y:noise_y + block_noise_size_y]
        # print(window)
        window = window.flatten()
        np.random.shuffle(window)
        window = window.reshape((1, block_noise_size_x,
                                 block_noise_size_y))
        image_temp[0, noise_x:noise_x + block_noise_size_x,
        noise_y:noise_y + block_noise_size_y] = window
    local_shuffling_x = image_temp
    # print('local_shuffling')
    return local_shuffling_x


def image_in_painting(x):
    img_deps, img_rows, img_cols = x.shape
    image_temp = copy.deepcopy(x)
    cnt = 5
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = random.randint(img_rows // 20, img_rows // 10)
        block_noise_size_y = random.randint(img_cols // 20, img_cols // 10)
        # block_noise_size_z = random.randint(img_deps//6, img_deps//3)
        noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
        noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
        # noise_z = random.randint(3, img_deps-block_noise_size_z-3)

        image_temp[0,
        noise_x:noise_x + block_noise_size_x,
        noise_y:noise_y + block_noise_size_y] = np.full((block_noise_size_x, block_noise_size_y),
                                                        np.random.rand(1)[0])
    # print('inpaint')    
    return image_temp


def image_out_painting(x):
    img_deps, img_rows, img_cols, = x.shape
    image_temp = copy.deepcopy(x)
    x = np.full((x.shape[0], x.shape[1], x.shape[2]), np.random.rand(1)[0])
    block_noise_size_x = img_rows - random.randint(3 * img_rows // 12, 4 * img_rows // 12)
    block_noise_size_y = img_cols - random.randint(3 * img_cols // 12, 4 * img_cols // 12)
    noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
    noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
    x[0,
    noise_x:noise_x + block_noise_size_x,
    noise_y:noise_y + block_noise_size_y, ] = image_temp[0, noise_x:noise_x + block_noise_size_x,
                                              noise_y:noise_y + block_noise_size_y]
    cnt = 4
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = img_rows - random.randint(3 * img_rows // 12, 4 * img_rows // 12)
        block_noise_size_y = img_cols - random.randint(3 * img_cols // 12, 4 * img_cols // 12)
        # block_noise_size_z = img_deps - random.randint(3*img_deps//7, 4*img_deps//7)
        noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
        noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
        # noise_z = random.randint(3, img_deps-block_noise_size_z-3)
        x[:,
        noise_x:noise_x + block_noise_size_x,
        noise_y:noise_y + block_noise_size_y] = image_temp[:, noise_x:noise_x + block_noise_size_x,
                                                noise_y:noise_y + block_noise_size_y]
    # print('out_paint')

    return x

def generate_pair(img, batch_size, config, status="test"):
    img_deps, img_rows, img_cols, = img.shape
    # Autoencoder
    x = copy.deepcopy(img)

    # Flip
    #x,y = data_augmentation(x,img, config.flip_rate)
    #print(config.local_rate,config.nonlinear_rate)
    # Local Shuffle Pixel
    x = local_pixel_shuffling(img, prob=config.local_rate)

    # Apply non-Linear transformation with an assigned probability
    x = nonlinear_transformation(x, config.nonlinear_rate)

    # Inpainting & Outpainting
    if random.random() < config.paint_rate:
        if random.random() < config.inpaint_rate:
            # Inpainting
            x = image_in_painting(x)
        else:
            # Outpainting
            x = image_out_painting(x)

    return x