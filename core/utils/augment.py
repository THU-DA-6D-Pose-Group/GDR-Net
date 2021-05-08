import random

import cv2
import numpy as np


class AugmentRGB(object):
    """Augmentation tool for detection problems.

    Parameters
    ----------

    brightness_var: float, default: 0.3
        The variance in brightness

    hue_delta: float, default: 0.1
        The variance in hue

    lighting_std: float, default: 0.3
        The standard deviation in lighting

    saturation_var: float, default: (0.5, 1.25)
        The variance in saturation

    contrast_var: float, default: (0.5, 1.25)
        The variance in constrast

    swap_colors: bool, default: False
        Whether color channels should be randomly flipped

    Notes
    -----
    Assumes images and labels to be in the range [0, 1] (i.e. normalized)

    All new operations are drafted from the TF implementation
    https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/python/ops/image_ops_impl.py

    Look here: https://github.com/weiliu89/caffe/blob/ssd/examples/ssd/ssd_pascal_resnet.py
    """

    def __init__(
        self,
        brightness_delta=32.0 / 255.0,
        hue_delta=0,
        lighting_std=0.3,
        saturation_var=(0.75, 1.25),
        contrast_var=(0.75, 1.25),
        swap_colors=False,
    ):

        # Build a list of color jitter functions
        self.color_jitter = []

        if brightness_delta:
            self.brightness_delta = brightness_delta
            self.color_jitter.append(self.random_brightness)
        if hue_delta:
            self.hue_delta = hue_delta
            self.color_jitter.append(self.random_hue)
        if saturation_var:
            self.saturation_var = saturation_var
            self.color_jitter.append(self.random_saturation)
        if contrast_var:
            self.contrast_var = contrast_var
            self.color_jitter.append(self.random_contrast)

        self.lighting_std = lighting_std
        self.swap_colors = swap_colors

    def augment(self, img):

        augment_type = np.random.randint(0, 2)
        if augment_type == 0:  # Take the image as a whole
            pass
        elif augment_type == 1:  # Random downsizing of original image
            pass  # img, lbl = self.random_rescale(img, lbl)

        random.shuffle(self.color_jitter)
        for jitter in self.color_jitter:
            img = jitter(img)
        return img

    def random_brightness(self, img):
        """Adjust the brightness of images by a random factor.

        Basically consists of a constant added offset.
        Args:
          image: An image.
        Returns:
          The brightness-adjusted image.
        """
        max_delta = self.brightness_delta
        assert max_delta >= 0
        delta = -max_delta + 2 * np.random.rand() * max_delta
        return np.clip(img + delta, 0.0, 1.0)

    def random_contrast(self, img):
        """For each channel, this function computes the mean of the image
        pixels in the channel and then adjusts each component.

        `x` of each pixel to `(x - mean) * contrast_factor + mean`.
        Args:
          image: RGB image or images. Size of the last dimension must be 3.
        Returns:
          The contrast-adjusted image.
        """
        lower, upper = self.contrast_var
        assert 0.0 <= lower <= upper
        contrast_factor = lower + 2 * np.random.rand() * (upper - lower)
        means = np.mean(img, axis=(0, 1))
        return np.clip((img - means) * contrast_factor + means, 0.0, 1.0)

    def random_saturation(self, img):
        """Adjust the saturation of an RGB image by a random factor.

        Equivalent to `adjust_saturation()` but uses a `saturation_factor` randomly
        picked in the interval `[lower, upper]`.
        Args:
          image: RGB image or images. Size of the last dimension must be 3.
        Returns:
          Adjusted image(s), same shape and DType as `image`.
        """
        lower, upper = self.saturation_var
        assert 0.0 <= lower <= upper
        saturation_factor = lower + 2 * np.random.rand() * (upper - lower)
        return self.adjust_saturation(img, saturation_factor)

    def random_hue(self, img):
        """Adjust the hue of an RGB image by a random factor.

        Equivalent to `adjust_hue()` but uses a `delta` randomly
        picked in the interval `[-max_delta, max_delta]`.
        `hue_delta` must be in the interval `[0, 0.5]`.
        Args:
          img: RGB image or images. Size of the last dimension must be 3.
        Returns:
          Numpy image
        """
        max_delta = self.hue_delta
        assert 0.0 <= max_delta <= 0.5
        delta = -max_delta + 2.0 * np.random.rand() * max_delta
        return self.adjust_hue(img, delta)

    def adjust_gamma(self, img, gamma=1.0, gain=1.0):
        """Performs Gamma Correction on the input image.
          Also known as Power Law Transform. This function transforms the
          input image pixelwise according to the equation Out = In**gamma
          after scaling each pixel to the range 0 to 1.
        Args:
          img : Numpy array.
          gamma : A scalar. Non negative real number.
          gain  : A scalar. The constant multiplier.
        Returns:
          Gamma corrected numpy image.
        Notes:
          For gamma greater than 1, the histogram will shift towards left and
          the output image will be darker than the input image.
          For gamma less than 1, the histogram will shift towards right and
          the output image will be brighter than the input image.
        References:
          [1] http://en.wikipedia.org/wiki/Gamma_correction
        """

        assert gamma >= 0.0
        # According to the definition of gamma correction
        return np.clip(((img ** gamma) * gain), 0.0, 1.0)

    def adjust_hue(self, img, delta):
        """Adjust hue of an RGB image.

        Converts an RGB image to HSV, add an offset to the hue channel and converts
        back to RGB. Rotating the hue channel (H) by `delta`.
        `delta` must be in the interval `[-1, 1]`.
        Args:
            image: RGB image
            delta: float.  How much to add to the hue channel.
        Returns:
            Adjusted image as np
        """
        assert img.shape[2] == 3
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # OpenCV returns hue from 0 to 360
        hue, sat, val = cv2.split(hsv)

        # Note that we add 360 to guarantee that the resulting hue is a positive
        # floating point number since delta is [-0.5, 0.5].
        hue = np.mod(360 + hue + delta * 360, 360.0)
        hsv_altered = cv2.merge((hue, sat, val))
        return cv2.cvtColor(hsv_altered, cv2.COLOR_HSV2BGR)

    def adjust_saturation(self, img, saturation_factor):
        """Adjust saturation of an RGB image.

        `image` is an RGB image.  The image saturation is adjusted by converting the
        image to HSV and multiplying the saturation (S) channel by
        `saturation_factor` and clipping. The image is then converted back to RGB.
        Args:
          img: RGB image or images. Size of the last dimension must be 3.
          saturation_factor: float. Factor to multiply the saturation by.
        Returns:
          Adjusted numpy image
        """

        assert img.shape[2] == 3
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)
        sat = np.clip(sat * saturation_factor, 0.0, 1.0)
        hsv_altered = cv2.merge((hue, sat, val))
        return cv2.cvtColor(hsv_altered, cv2.COLOR_HSV2BGR)

    def swap_colors(self, img):
        """Randomly swap color channels."""
        # Skip swapping?
        if np.random.random() > 0.5:
            return img

        swap = np.random.randint(5)
        if swap == 0:
            img = 1.0 - img
        elif swap == 1:
            img = img[:, :, [0, 2, 1]]
        elif swap == 2:
            img = img[:, :, [2, 0, 1]]
        elif swap == 3:
            img = img[:, :, [1, 0, 2]]
        elif swap == 4:
            img = img[:, :, [1, 2, 0]]
        return img

    def grayscale(self, rgb):
        return rgb.dot([0.299, 0.587, 0.114])

    def saturation(self, rgb):
        """Randomly change saturation."""
        gs = self.grayscale(rgb)
        alpha = 2 * np.random.random() * self.saturation_var
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
        return np.clip(rgb, 0, 1)

    def brightness(self, rgb):
        """Randomly change brightness."""
        alpha = 2 * np.random.random() * self.brightness_var
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha
        return np.clip(rgb, 0, 1)

    def contrast(self, rgb):
        """Randomly change contrast levels."""
        gs = self.grayscale(rgb).mean() * np.ones_like(rgb)
        alpha = 2 * np.random.random() * self.contrast_var
        alpha += 1 - self.contrast_var
        rgb = rgb * alpha + (1 - alpha) * gs
        return np.clip(rgb, 0, 1)

    def lighting(self, img):
        """Randomly change lighting."""
        cov = np.cov(img.reshape(-1, 3), rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        noise = np.random.randn(3) * self.lighting_std
        noise = eigvec.dot(eigval * noise)
        img += noise
        return np.clip(img, 0, 1)
