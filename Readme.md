## How to use

### Step 1: Environment
- Install our environment using the provided environment.yml 
- In the package "flax", find the file "flax/linen/normalization.py" and comment out line 362

### Step 2: Run experiments
- Run the file "run_all.sh"

### Step 3: Make plots
- Run the .ipynb notebooks in "./plots"


## Documentation

There are 3 major folders. "./code" contains the code neccessary for running the experiments. "./data" contains the dataset, as well as model weights, gradients, stats etc. Each experiment has a settings.json, which sets the neccessary variables. "./plots" contains code neccessary for plotting as well as ".iypnb" notebooks which generate the plots

### Code subdirectory
- utils.py contains most of the functions neccessary for running
- models.py contains the VGG11
- main.py is the file that is called in order to run the experiments. It expects the path to a settings.json as an argument
- consecutive_grads.py loads a specific checkpoint and saves a series of checkpoints in a separate folder. The difference to main.py is that nothing besides the models states and gradients is saved.
- distribution_drift.py loads a specific consecutive_grads.py folder and calculates the ICS for each checkpoint
- adam_drift.py loads a specific consecutive_grads.py saving folder and calculates the cosine similarity between adam's running stats and our estimation
- sgdm_drift.py loads a specific consecutive_grads.py saving folder and calculates the ICS for each checkpoint sgdm's running stats and our estimation