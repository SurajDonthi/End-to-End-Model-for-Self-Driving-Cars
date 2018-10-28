# End-to-End-Model-for-Self-Driving-Cars
An attempt to run Self-Driving cars using an end-to-end Deep Learning Model.

This model is inspired by Nvidia's [End to End Learning for Self-Driving Cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)[1]. The DNN model created here has some modificaitons over the original network by including the acceleration also as the output parameter.

## The Deep Learning Model
![Alt text]()

## Datasets
The following Datasets by Udacity are appropriate and in full relevance to the Deep Learning Model presented here. However, the raw sensory data are stored in `rosbag (.bag)` format. It needs to extracted using the ROS Python API. A full description of the procedure for extracting the data can be found in the 'Datasets' directory.

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
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
  <tr>
    <td> Udacity Driving Challenge 2 - 02
    <td align="center"> -
    <td align="center">
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
  <tr>
    <td> Udacity Driving Challenge 3 - 01
    <td align="center"> -
    <td align="center">
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
  <tr>
    <td> Udacity Driving Challenge 3 - 02
    <td align="center"> -
    <td align="center">
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
    <td align="center"><ul><li>[ ] </ul>
  </tr>
</table>
      
A more detailed list of other available driving datasets can be found from this [paper](https://ieeexplore.ieee.org/document/8317828)[3].

### Data Exploration

### Data Preprocessing
#### 1. Removing skewness in the data.

## Data Augmentation
The Data Augmentation techniques are inspired from the [paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)[1] and this [blog](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9)[2].

#### 1. Random shift.

#### 2. Random brighness.

#### 3. Random Shadow.

#### 4. Random Flip.

#### 5. Random Translate.

## Network Architecture
![Alt text]()

### Training

## Future Work
- [ ] Merge Object Detection/Segmentation Models(YOLO/SSD/Faster-RCNN/Mask-RCNN etc.) into the network.
- [ ] Utilize Transfer Learning for better Accuracy
- [ ] Change Keras model to Tensorflow Model
- [ ] Integrate other parameters like Speed, Point Cloud Data into the network

## References
[1] https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf <br>
[2] https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9 <br>
[3] https://ieeexplore.ieee.org/document/8317828
