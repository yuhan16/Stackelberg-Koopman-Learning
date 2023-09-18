# Stackelberg Game and Koopman Operator
This repo is for SG Koopman Learning project with an application in the guided navigation (trajectory planning).

## Requirements
- Python 3.11

## Running Scripts
1. Create a python virtual envrionment with Python 3.11 and source the virtual environment:
```bash
$ python3.11 -m venv <your-virtual-env-name>
$ source /path-to-venv/bin/activate
```
2. `pip` install the requirements:
```bash
$ pip install -r requirements.txt
```
3. In the project directory, first run the following script to generate data:
```bash
$ python exp_scripts/generate_data.py
```
4. In the project directory, run other scripts by commenting or uncommenting related functions:
```bash
$ python exp_scripts/training.py   # run training.py with train_kp_nn() as an example
```
5. (Optional) Create log directory and generate logs files by redicecting the output:
```bash
$ mkdir logs
$ mkdir logs/kp_nn         # example log directory
$ python exp_scripts/training.py > logs/kp_nn/log.txt
```


## Comparison Algorithms
- `kp_nn`: Use Koopman operator to learn the follower's feedback dynamics.
- `nn_fdynbr`: Use neural network to learn the follower's feedback dynamics.
- `dmd`: Use DMD to learn the follower's feedback dynamics.
- `nonlin_ocp`: Use nonlinear optimal control (modle-based) to solve SE, serve as the baseline.


## File Structure
- `sg_koopman`: Full algorithm implementations.
    - `sg_koopman/common`: Common classes and utilities.
- `exp_scripts`: Examples of calling modules and functions in `sg_koopman`.
    - Each script performs the function as the name suggests. E.g., `training.py` performs training for all algorithms.
    - The training (and RH planning) for each comparison algorithm is encapsulated by a function in each script. E.g., `train_kp_nn()` implements training for `kp_nn` algorithm.
    - Comment or uncomment functions to run each comparison algorithm.
- `data`: Store learning and planning results. 
    - Each algorithm has an independent directory.


## Coding Specifications
Individual agent's trajectory data is stored in a 3D numpy array with the format: `D[i, k, :] = [x_k, u_k, x_kp1]`.
- `D.shape[0]` is the total number of trajectory.
- `D.shape[1]` is the length of each trajectory.

**Note:** `x_kp1` in `D[i,k,:]` is the same as `x_k` in `D[i,k+1,:]`. We use the redundancy to align the data for fast access.

Interactive trajectory data is stored in a 3D numpy array with the format: `D[i, k, :] = [xf_k, ufopt_k, xf_kp1, xl_k, ul_k, xl_kp1]`.
- The follower's control `ufopt_k` is optimal w.r.t. the leader's trajectory.
- Use the same data redundancy to align the data.