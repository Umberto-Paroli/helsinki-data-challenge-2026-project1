# Helsinki Data Challenge 2026 - Team UNIMORE - Solution 1

## Installation
It is suggested the use of a virtual environment or a container to manage project dependencies.
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
## Data
Before run the code add asteroids lightcurves and .stl of the public objects in [Data directory](./Data/). Video and blender renderings arent required. 

## Usage
To run the full pipeline for an unknown object fix the lightcurves path according to the desired object in the following configuration files:
- [config_calibrate_unknown](./config_calibrate_unknown.yaml)
- [config_convex_unknown](./config_convex_unknown.yaml)
- [config_refine_unknown](./config_refine_unknown.yaml)

Then run:
```bash
python ./run_unknown_object.py
```

Hyperparameters of the process are obtained trough [search.py](./search.py) which optimizes over the Public Shapes of the data challenge (see files comments for more details).

## Logic
This project, which extend the first solution code, divide the problem into two phases.\
First it performs a **convex reconstruction** where an initial shape (an ellipsoid) is adapted to the lightcurves of an object. \
Second, a **refinement** step is applied to retrieve non-convex features of the unknown objects.\
Each step is performed by an independent model trained with challenge publics data trough a forward operator that emulate the process to obtain lightcurves.

For more details non model architectures and training process see [train_convex.py](./train_convex.py) and [train.py](./train_convex.py). \
**Note:** In those files device is set to "cuda:0". Change to "cuda:1" to allow the usage of GPU, if available.
