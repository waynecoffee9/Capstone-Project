## Work In Progress

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
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

## Status
The car can run and stop at red light with ground truth red light signal from the simulator.  Next step is to train a model to classify the lights.
