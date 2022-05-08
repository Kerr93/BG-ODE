# Prepare
* run prepare.py in ```real``` folder
* run gen_sim_data.py in ```simulation``` folder

# Run Experiments
* run expert model (please run expert model first)
  * ```python main.py --exp expert```
* run neural model
  * ```python main.py --exp neural```
* neural model with augmented data
  * ```python main.py --exp data_aug```
* tune expert model on the real-world data
  * ```python main.py --exp union```

# Train with partial data
You may use --frac augmentation to train the model with partial data: use --frac 0.1 to random sample 10% data; use --frac 540 to filter out the patient 540.

# BG-ODE
