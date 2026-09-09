# Helsinki Data Challenge 2026 - Team UNIMORE - Solution 1

## Installation
It is suggested the use of a virtual environment or a container to manage project dependencies.
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
## Data
Before run the code add asteroids lightcurves and .stl of the public objects in a **./Data** directory. Real videos and blender renderings aren't required. 

## Usage
Running [search.py](./search.py) find a best hyperparameter configuration for the three public objects available from th data challenge. \
The resulting values has to be replicated in configuration files for each unknown objects but changing the light curve file path adn the cylinder_radius which are different for every objects. \
The configuration files present in the repository are obtained by running:
```bash
python search.py --phase convex --hours 24
python search.py --phase refine --hours 24
```
In each configuration file one must set:
```YAML
  data:
    intensity_file: <LC_INTENSITY_PATH>
    binary_file: <LC_INTENSITY_BINARY>
    intensity_file_blender: <LC_BLENDER_INTENSITY_PATH> # only if this object has published Blender data
    binary_file_blender: <LC_BLENDER_BINARY_PATH>
```
Change also output directory if required
```YAML
  output:
    base_dir: ./runs_unknown/[N] #if desired
```

When the configuration files are ready we can run the three stage of the process in sequence:
```bash
python ./run_unknown_object.py
```

We can also manually call each script, but doing so requires to specify the checkpoint for the refinement step in the configuration files
```bash
python main_convex.py ./config_calibrate_unknown.yaml
python main_convex.py ./config_convex_unknown.yaml
python main.py ./config_refine_unknown.yaml
```

## Saved Configuratio files
To simplify the work and avoid running the search of hyperparameters we include our configuration files for each unknown objects in [configs_unknown directory](./configs_unknown/).

## Project summary
This project, which extend the first solution code, divide the problem into two phases.\
First it performs a **convex reconstruction** where an initial shape (an ellipsoid as default) is adapted to the lightcurves of an object. \
Second, a **non-convex refinement** step is applied to retrieve features lost from a convex approximation of the unknown objects.\
Each step is performed by an independent model trained with challenge publics data trough a forward operator that emulates lightcurves experimental setup.


[INSERT HERE SOME LINES ABOUT THE MODELS]

For more details on model architectures and training process see [train_convex.py](./train_convex.py) and [train.py](./train_convex.py). \
**Note:** In those files device is set to "cuda:0". Change to "cuda:1" to allow the usage of GPU, if available.
