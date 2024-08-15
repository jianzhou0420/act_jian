import numpy as np
import re
from PIL import Image

DEFAULT_RGB_SCALE_FACTOR = 2**24 - 1  # JIAN: note this is actually the scale factor that rlbench uses not its default 2560000
DEFAULT_GRAY_SCALE_FACTOR = {np.uint8: 100.0,
                             np.uint16: 1000.0,
                             np.int32: DEFAULT_RGB_SCALE_FACTOR}


def float_array_to_rgb_image(float_array,
                             scale_factor=DEFAULT_RGB_SCALE_FACTOR,
                             drop_blue=False):
    '''
    Jian: From RLBench: https://github.com/stepjam/RLBench/blob/master/rlbench/backend/utils.py
    '''
    """Convert a floating point array of values to an RGB image.

    Convert floating point values to a fixed point representation where
    the RGB bytes represent a 24-bit integer.
    R is the high order byte.
    B is the low order byte.
    The precision of the depth image is 1/256 mm.

    Floating point values are scaled so that the integer values cover
    the representable range of depths.

    This image representation should only use lossless compression.

    Args:
      float_array: Input array of floating point depth values in meters.
      scale_factor: Scale value applied to all float values.
      drop_blue: Zero out the blue channel to improve compression, results in 1mm
        precision depth values.

    Returns:
      24-bit RGB PIL Image object representing depth values.
    """
    def ClipFloatValues(float_array, min_value, max_value):
        """Clips values to the range [min_value, max_value].

        First checks if any values are out of range and prints a message.
        Then clips all values to the given range.

        Args:
        float_array: 2D array of floating point values to be clipped.
        min_value: Minimum value of clip range.
        max_value: Maximum value of clip range.

        Returns:
        The clipped array.

        """
        if float_array.min() < min_value or float_array.max() > max_value:
            float_array = np.clip(float_array, min_value, max_value)
        return float_array

    # Scale the floating point array.
    scaled_array = np.floor(float_array * scale_factor + 0.5)
    # Convert the array to integer type and clip to representable range.
    min_inttype = 0
    max_inttype = 2**24 - 1
    scaled_array = ClipFloatValues(scaled_array, min_inttype, max_inttype)
    int_array = scaled_array.astype(np.uint32)
    # Calculate:
    #   r = (f / 256) / 256  high byte
    #   g = (f / 256) % 256  middle byte
    #   b = f % 256          low byte
    rg = np.divide(int_array, 256)
    r = np.divide(rg, 256)
    g = np.mod(rg, 256)
    image_shape = int_array.shape
    rgb_array = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
    rgb_array[..., 0] = r
    rgb_array[..., 1] = g
    if not drop_blue:
        # Calculate the blue channel and add it to the array.
        b = np.mod(int_array, 256)
        rgb_array[..., 2] = b
    image_mode = 'RGB'
    image = Image.fromarray(rgb_array, mode=image_mode)
    return image


def image_to_float_array(image, scale_factor=None):
    '''
    Jian: From RLBench: https://github.com/stepjam/RLBench/blob/master/rlbench/backend/utils.py
    seems it need the image to have shape [H,W,3]
    '''

    """Recovers the depth values from an image.

    Reverses the depth to image conversion performed by FloatArrayToRgbImage or
    FloatArrayToGrayImage.

    The image is treated as an array of fixed point depth values.  Each
    value is converted to float and scaled by the inverse of the factor
    that was used to generate the Image object from depth values.  If
    scale_factor is specified, it should be the same value that was
    specified in the original conversion.

    The result of this function should be equal to the original input
    within the precision of the conversion.

    Args:
    image: Depth image output of FloatArrayTo[Format]Image.
    scale_factor: Fixed point scale factor.

    Returns:
    A 2D floating point numpy array representing a depth image.

    """
    image_array = np.array(image)
    image_dtype = image_array.dtype
    image_shape = image_array.shape

    channels = image_shape[2] if len(image_shape) > 2 else 1
    assert 2 <= len(image_shape) <= 3
    if channels == 3:
        # RGB image needs to be converted to 24 bit integer.
        float_array = np.sum(image_array * [65536, 256, 1], axis=2)
        if scale_factor is None:
            scale_factor = DEFAULT_RGB_SCALE_FACTOR
    else:
        if scale_factor is None:
            scale_factor = DEFAULT_GRAY_SCALE_FACTOR[image_dtype.type]
        float_array = image_array.astype(np.float32)
    scaled_array = float_array / scale_factor
    return scaled_array


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
