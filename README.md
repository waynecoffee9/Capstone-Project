# Udacity Self-Driving Car Capstone Project


## The TEAM

1. Kibaek Jeong
2. Danfeng Xu
3. Roopak Ingole
4. Ying Tang
5. Wayne Chen

### Usage

1. Clone the project repository

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```

3. Download necessary model files if they are missing from folder:

\ros\src\tl_detector\light_classification\model

[frozen_inference_graph_mobilenet.pb](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz)

[frozen_inference_graph_rfcn_resnet101.pb](http://download.tensorflow.org/models/object_detection/rfcn_resnet101_coco_11_06_2017.tar.gz)

Extract the .pb files to the specified folder, and rename them as above.

4. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

## Status
The car can run and stop at red light with ground truth red light signal from the simulator.  Next step is to train a model to classify the lights.
