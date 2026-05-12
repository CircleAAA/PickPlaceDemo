# Pick & Place Demo

This simple project is to show how to train a multi-frame-input and multi-frame-prediction vision-action model to complete a two-stage task. The first stage is to reach and pick a cube on the ground. The second stage is then to carry the object to the target place. The stage is pivoted 

We simulate the task in Blender. The robot arm is simplified as a red square column, placed on the green floor. The object to be caught is simplified as a blue cube. The target place is yellow-colored square area in the floor.

We use multi-frame-input and multi-frame-prediction model to fit polyline trajectory, and binary cross entropy loss to fit the binary IO signal which controls the end effector.

## Setup Environment

```bash
# create python virtual environment
py -3.11 -m venv venv

# activate python virtual environment on windows
source venv/Script/activate

# Python 3.11.0
python --version

# install required packages, including blender as a python module.
# See https://docs.blender.org/api/current/info_advanced_blender_as_bpy.html
bash install.sh
```

## Generate Synthesis Data
```bash
python record_episode.py --num_episodes 500 --resolution 128
```

Then prepare the train and validate dataset.
```bash
python generate_data.py
```

## Training
```bash
python train.py
```

## Test Online
Here "online" means the deep learning model interacts dynamically with the virtual 3D scene.
```bash
python test.py
```