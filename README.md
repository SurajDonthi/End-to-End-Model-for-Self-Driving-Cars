# End-to-End-Model-for-Self-Driving-Cars
An attempt to run Self-Driving cars using an end-to-end Deep Learning Model.

This model is inspired by Nvidia's [End to End Learning for Self-Driving Cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)[1]. The DNN model created here has some modificaitons over the original network by including the throttle, speed, brake and Point Cloud Data into consideration.

## The Deep Learning Model
<p align="center">
  <img src="https://github.com/SurajDonthi/End-to-End-Model-for-Self-Driving-Cars/blob/master/Doc_Images/Model%20Architecture.JPG"><br>
  <i>The base model</i>
</p>

The model network consists of preprocessin steps and is fed into a 10 layer CNN. The output which is the steering angle is trained by back propagation.

## Datasets
The following Datasets by Udacity are appropriate and in full relevance to the Deep Learning Model presented here. However, the raw sensory data are stored in `rosbag (.bag)` format. It needs to extracted using the ROS Python API. A full description of the procedure for extracting the data can be found in the 'Datasets' directory.

#### Download any one of the datasets for training.

<table>
  <tr>
    <td rowspan=2 align="center"><h3>Datasets
    <td rowspan=2 align="center"><h3>Duration 
    <td rowspan=2 align="center"><h3>Size 
    <td rowspan=1 colspan=10 align="center"><h3>Parameters
  <tr> 
    <td align="center"><h4>Center Image	
    <td align="center"><h4>Right Image 
    <td align="center"><h4>Left Image
    <td align="center"><h4>Lattitude & Longitude
    <td align="center"><h4>Steering Angle 
    <td align="center"><h4>Throttle
    <td align="center"><h4>Brake
    <td align="center"><h4>Speed
    <td align="center"><h4>Gear
    <td align="center"><h4>Point Cloud Data
  <tr>
    <td> Udacity Driving Challenge 2 - 01
    <td align="center"> -
    <td align="center">
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[ ] </ul>
  <tr>
    <td> Udacity Driving Challenge 2 - 02
    <td align="center"> -
    <td align="center">
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[ ] </ul>
  <tr>
    <td> Udacity Driving Challenge 3 - 01
    <td align="center"> -
    <td align="center">
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[x] </ul>
  <tr>
    <td> Udacity Driving Challenge 3 - 02
    <td align="center"> -
    <td align="center">
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[x] </ul>
    <td align="center"><ul><li>[x] </ul>
  </tr>
</table>
      
A more detailed list of other available driving datasets can be found from this [paper](https://ieeexplore.ieee.org/document/8317828)[3].

### Data Exploration
The results & study of data exploration will be uploaded shortly.

### Data Preprocessing
#### 1. Removing skewness in the data.
The driving data usually contains moving on straight roads and hence can be found to have a large set of images with steering angle being almost 0. Likewise, the usage of extreme steering angles is also very low  and hence have very less images. This skewness in the data will not help the model to be trained well for driving on curved roads. We can encounter this problem by augmenting data and thereby creating an uniform distribution of the data.

## Data Augmentation
The Data Augmentation techniques are inspired from the [paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)[1] and this [blog](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9)[2].

#### 1. Randomly choose image and  shift steering angle
> One of the center, right or left images are randomly chosen and the steering angle is adjusted accordingly.

    def random_choose(data_dir, center, left, right, steering_angle):
        choice = np.random.choice(3)
        if choice == 0:
            return load_image(data_dir, left), steering_angle + 0.2
        elif choice == 1:
            return load_image(data_dir, right), steering_angle - 0.2
        return load_image(data_dir, center), steering_angle


#### 2. Random brightness
> Brightness is randomly added to the image by converting the RGB image to HSV within a range of 80% to 120%. the code for the same is as below:

    def random_brightness(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
        hsv[:, :, 2] = hsv[:, :, 2] * ratio
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

#### 3. Random Shadow
> The random shadow function creates a random patch of shadow in the image.

    def random_shadow(image):
    
        # Randomly create (x1, y1) and (x2, y2) to create a patch area
        x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
        x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
        xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]
        
        # Create a mask to create a patch
        mask = np.zeros_like(image[:, :, 1])
        mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

        # choose which side should have shadow and adjust saturation
        cond = mask == np.random.randint(2)
        s_ratio = np.random.uniform(low=0.2, high=0.5)

        # adjust Saturation in HLS(Hue, Light, Saturation)
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
        return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

#### 4. Random Flip
> The images are randomly flipped with a 50% probability and the sttering angle in negated to account for the changes.

    def random_flip(image, steering_angle):
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
            steering_angle = -steering_angle
        return image, steering_angle

#### 5. Random Translate
>The image is randomly translated/shifted and the necessary steering angle adjustments are made.

    def random_translate(image, steering_angle, range_x, range_y):
        trans_x = range_x * (np.random.rand() - 0.5)
        trans_y = range_y * (np.random.rand() - 0.5)
        steering_angle += trans_x * 0.002
        trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        height, width = image.shape[:2]
        image = cv2.warpAffine(image, trans_m, (width, height))
        return image, steering_angle


## Network Architecture
<p align="center">
  <img src="https://github.com/SurajDonthi/End-to-End-Model-for-Self-Driving-Cars/blob/master/Doc_Images/Network%20Architecture.JPG"><br>
  <i>The base network architecture</i>
</p>

 - The network consists of a 11 layer(including the input) Convolutional Neural Network as shown in the above figure.
 - The input is first normalised between -0.5 and 0.5 and then fed to the network.
 - The first 3 layers use a 5 ![symbol](http://latex.codecogs.com/gif.latex?\times) 5 kernels and the next 2 are 3 ![symbol](http://latex.codecogs.com/gif.latex?\times) 3 kernels.
 - The next layer is flattened out to form a fully connected layer(FCN).
 -  As in the image the FCNs are reduced to 100, 50, 10 and lastly 1.
 - The output is the steering angle.

## Future Work
- [ ] Merge Object Detection/Segmentation Models(YOLO/SSD/Faster-RCNN/Mask-RCNN etc.) into the network.
- [ ] Utilize Transfer Learning for better Accuracy
- [ ] Change Keras model to Tensorflow Model
- [ ] Integrate other parameters like Speed, Point Cloud Data into the network

## References
[1] https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf <br>
[2] https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9 <br>
[3] https://ieeexplore.ieee.org/document/8317828
