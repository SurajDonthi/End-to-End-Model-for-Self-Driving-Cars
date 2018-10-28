import cv2
import os
import numpy as np

# <editor-fold desc=" TODO: Create a Load & ProcessImage Class">
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def load_image(data_dir, image_file):                   # 3 Function calls @ batch_generator & 2 * choose_image
    """
    Load RGB images from a file
    """
    return cv2.imread(os.path.join(data_dir, image_file.strip()))


def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = image[60:-25, :, :]
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image
# </editor-fold>


class AugData(list):

    def __new__(cls, data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
        """
        Generate an augumented image and adjust steering angle.
        (The steering angle is associated with the center image)

        __new__ uses 5 functions:
            1. choose_image ----> load_image
            2. random_flip
            3. random_translate
            4. random_shadow
            5. random_brightness
        """

        image, steering_angle = cls.random_choose(data_dir, center, left, right, steering_angle)
        image, steering_angle = cls.random_flip(image, steering_angle)
        image, steering_angle = cls.random_translate(image, steering_angle, range_x, range_y)
        image = cls.random_shadow(image)
        image = cls.random_brightness(image)
        return super().__new__(cls, [image, steering_angle])

    # <editor-fold desc="TODO: Create a Data Augmentation Class">
    def random_choose(data_dir, center, left, right, steering_angle):
        """
        Randomly choose an image from the center, left or right, and adjust
        the steering angle.
        """
        choice = np.random.choice(3)
        if choice == 0:
            return load_image(data_dir, left), steering_angle + 0.2
        elif choice == 1:
            return load_image(data_dir, right), steering_angle - 0.2
        return load_image(data_dir, center), steering_angle

    def random_flip(image, steering_angle):
        """
        Randomly flipt the image left <-> right, and adjust the steering angle.
        """
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
            steering_angle = -steering_angle
        return image, steering_angle

    def random_translate(image, steering_angle, range_x, range_y):
        """
        Randomly shift the image virtially and horizontally (translation).
        """
        trans_x = range_x * (np.random.rand() - 0.5)
        trans_y = range_y * (np.random.rand() - 0.5)
        steering_angle += trans_x * 0.002
        trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        height, width = image.shape[:2]
        image = cv2.warpAffine(image, trans_m, (width, height))
        return image, steering_angle

    def random_shadow(image):
        """
        Generates and adds random shadow
        """
        # (x1, y1) and (x2, y2) forms a line
        # xm, ym gives all the locations of the image
        x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
        x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
        xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

        # mathematically speaking, we want to set 1 below the line and zero otherwise
        # Our coordinate is up side down.  So, the above the line:
        # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
        # as x2 == x1 causes zero-division problem, we'll write it in the below form:
        # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
        mask = np.zeros_like(image[:, :, 1])
        mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

        # choose which side should have shadow and adjust saturation
        cond = mask == np.random.randint(2)
        s_ratio = np.random.uniform(low=0.2, high=0.5)

        # adjust Saturation in HLS(Hue, Light, Saturation)
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
        return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

    def random_brightness(image):
        """
        Randomly adjust brightness of the image.
        """
        # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
        hsv[:, :, 2] = hsv[:, :, 2] * ratio
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


    # </editor-fold>


def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    """
    Generate training image give image paths and associated steering angles
    """
    "This is the only function used in other files. The rest can be used within the class."
    """
    This function uses 3 other functions:
        1. load_image()
        2. preprocess()
        3. augment()
    """

    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            # argumentation
            if is_training and np.random.rand() < 0.6:
                image, steering_angle = AugData(data_dir, center, left, right, steering_angle)
            else:
                image = load_image(data_dir, center)
            # add the image and steering angle to the batch
            images[i] = preprocess(image)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers
