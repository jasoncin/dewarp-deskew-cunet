import os 
import cv2
import numpy as np
from scipy import misc
from scipy.ndimage import affine_transform as scipy_affine_transform
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


def to_categorical(target_vector, n_labels):
    return np.eye(n_labels, dtype='B')[target_vector]


def to_categorical_multi(target_vector, n_labels):
    return np.vstack((np.zeros((1, n_labels), dtype='B'), np.eye(n_labels, dtype='B')))[target_vector]


def read_image_list_path(path):
    return [os.path.join(path, f) for f in os.listdir(path)]


def calcAffineMatrix(sourcePoints, targetPoints):
    # For three or more source and target points, find the affine transformation
    A = []
    b = []
    for sp, trg in zip(sourcePoints, targetPoints):
        A.append([sp[0], 0, sp[1], 0, 1, 0])
        A.append([0, sp[0], 0, sp[1], 0, 1])
        b.append(trg[0])
        b.append(trg[1])
    result, resids, rank, s = np.linalg.lstsq(np.array(A), np.array(b))

    a0, a1, a2, a3, a4, a5 = result
    affineTrafo = np.float32([[a0, a2, a4], [a1, a3, a5]])
    return affineTrafo


def affine_transform(image, affine_value):
    shape = image.shape
    alpha_affine = min(shape[0], shape[1]) * affine_value
    random_state = np.random.RandomState(None)
    # Random affine
    shape_size = shape[:2]
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32(
        [center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
         center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = calcAffineMatrix(pts1, pts2)
    R = M[0:2, 0:2]
    Off = M[:, 2]
    for aD in range(shape[2]):
        image[:, :, aD] = scipy_affine_transform(image[:, :, aD], R, offset=Off)
    return image


def elastic_transform(image, elastic_value_x, elastic_value_y):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications JUST in Y-DIRECTION).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    shape = image.shape
    random_state = np.random.RandomState(None)
    nY = shape[0] // 25
    nX = shape[1] // 25
    sigma = min(shape[1], shape[0]) * 0.0025
    alpha_X = elastic_value_x * min(shape[0], shape[1])
    alpha_Y = elastic_value_y * min(shape[0], shape[1])
    dx = gaussian_filter((random_state.rand(nY, nX) * 2 - 1), sigma)
    dy = gaussian_filter((random_state.rand(nY, nX) * 2 - 1), sigma)
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    dx = misc.imresize(dx, [shape[0], shape[1]], interp='bicubic')
    dy = misc.imresize(dy, [shape[0], shape[1]], interp='bicubic')
    dxT = []
    dyT = []
    for dummy in range(shape[2]):
        dxT.append(dx)
        dyT.append(dy)
    dx = np.dstack(dxT)
    dy = np.dstack(dyT)
    dx = dx * alpha_X
    dy = dy * alpha_Y
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    image = map_coordinates(image, indices, order=1).reshape(shape)
    return image


def resize_width(image, new_w):
    h, w = image.shape[:2]
    # resize big image to preferred width
    if w > new_w:
        h = int(h * new_w / w)
        image = cv2.resize(image, (new_w, h), interpolation=cv2.INTER_AREA)
    return image


def resize_height(image, new_h):
    h, w = image.shape[:2]
    # resize big image to preferred heigth
    if h > new_h:
        w = int(w * new_h / h)
        image = cv2.resize(image, (w, new_h), interpolation=cv2.INTER_AREA)
    return image
