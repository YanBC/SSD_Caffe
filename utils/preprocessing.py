import caffe
import numpy as np
from scipy.ndimage import zoom
from skimage.transform import resize
import cv2 as cv



transformer = caffe.io.Transformer({'data': (1, 3, 300, 300)})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104, 117, 123]))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))


def resize_image(im, new_dims, interp_order=1):
    """
    Resize an image array with interpolation.

    Parameters
    ----------
    im : (H x W x K) ndarray
    new_dims : (height, width) tuple of new dimensions.
    interp_order : interpolation order, default is linear.

    Returns
    -------
    im : resized ndarray with shape (new_dims[0], new_dims[1], K)
    """
    if im.shape[-1] == 1 or im.shape[-1] == 3:
        im_min, im_max = im.min(), im.max()
        if im_max > im_min:
            # skimage is fast but only understands {1,3} channel images
            # in [0, 1].
            im_std = (im - im_min) / (im_max - im_min)
            resized_std = resize(im_std, new_dims, order=interp_order)
            resized_im = resized_std * (im_max - im_min) + im_min
        else:
            # the image is a constant -- avoid divide by 0
            ret = np.empty((new_dims[0], new_dims[1], im.shape[-1]),
                           dtype=np.float32)
            ret.fill(im_min)
            return ret
    else:
        # ndimage interpolates anything but more slowly.
        scale = tuple(np.array(new_dims, dtype=float) / np.array(im.shape[:2]))
        resized_im = zoom(im, scale + (1,), order=interp_order)
    return resized_im.astype(np.float32)



def caffe_proc(image):
    processed = transformer.preprocess('data', image)
    return processed


def numpy_proc(image):
    floated = image.astype(np.float32, copy=False)
    # resized = resize_image(floated, (300,300))
    # resized = np.resize(floated, (300,300,3))
    resized = cv.resize(floated, (300,300))
    swapped = resized[:,:,(2,1,0)]
    raw_scaled = 255.0 * swapped
    # minus = np.ones((300,300,3), dtype=np.float32) * np.array([104.0, 117.0, 123.0])
    minus = np.ones((300,300,3), dtype=np.float32)
    minus[:,:,0] *= 104.0
    minus[:,:,1] *= 117.0
    minus[:,:,2] *= 123.0
    meaned = raw_scaled - minus
    transposed = np.transpose(meaned, (2,0,1))
    return transposed



if __name__ == '__main__':
    image = np.random.rand(720,1080,3)

    caffe_image = caffe_proc(image)
    numpy_image = numpy_proc(image)
    # print(caffe_image == numpy_image)
    print(np.sum(np.abs(caffe_image - numpy_image)) / np.sum(caffe_image))



    import time

    start = time.time()
    for i in range(100):
        caffe_image = caffe_proc(image)
    print(time.time() - start)

    start = time.time()
    for i in range(100):
        numpy_image = numpy_proc(image)
    print(time.time() - start)
    

