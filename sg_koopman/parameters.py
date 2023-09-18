"""
This module defines all the parameters.
"""
import numpy as np
import torch
torch.set_default_dtype(torch.float64)


# environment settings
seed = 7096
rng = np.random.default_rng(seed)
torch.manual_seed(seed)
ws_len = 10
obs_spec = [
    [2.5,2.8,1, np.inf,0.5,1.2],    # [x,y,r, l_norm,x_scale,y_scale]
    [7,2,1,     2,1,1], 
    [2,7,1,     2,1,1], 
    [6,8,1,     1,1,1]
]
dxdy = 0.25     # resolution of grid map
def generate_grid_map(dxdy):
    '''
    Generates a grid map of the working space given the precision dxdy.
    coordinates: [x,y] = [i*dxdy, j*dxdy]
    '''
    def obs_detection(p): 
        for i in range(len(obs_spec)):
            obs_i = obs_spec[i]
            pxc, pyc, rc = obs_i[0], obs_i[1], obs_i[2]
            px_scale, py_scale = obs_i[4], obs_i[5]
            f1 = np.diag([1/px_scale, 1/py_scale]) @ (p-np.array([pxc,pyc]))
            if np.linalg.norm(f1, ord=obs_i[3]) <= rc:
                return True
        return False

    n_cell = int(ws_len / dxdy)
    grid_map = np.zeros((n_cell, n_cell), dtype=int)
        
    # brute force, test each cell to determine obstacles
    for i in range(n_cell):
        for j in range(n_cell):
            corners = [[i*dxdy, j*dxdy], [(i+1)*dxdy, j*dxdy], [i*dxdy, (j+1)*dxdy], [(i+1)*dxdy, (j+1)*dxdy]]
            for p in corners:
                if obs_detection(np.array(p)):
                    grid_map[i,j] = 1
                    break
        
    return grid_map.tolist()    # return a list
grid_map = generate_grid_map(dxdy)


# for dynamics
dimxl, dimul = 4, 2
dimxf, dimuf = 4, 2
dt = 0.2
dimT = int(1 / dt)
vmin, vmax = 0., 2.
wmin, wmax = -2., 2,
xd = np.array([9,9, np.cos(0), np.sin(0)])    # target state

xl0 = np.array([1, 2.5, np.cos(1.5), np.sin(1.5)])    # [px, py, theta]
xf0 = np.array([0.5, 3, np.cos(0.5), np.sin(0.5)])

xl0 = np.array([6, 0.5, np.cos(2.36), np.sin(2.36)])    # [px, py, theta]
xf0 = np.array([5.5, 0.1, np.cos(3), np.sin(3)])

xl0 = np.array([1, 8, np.cos(1.), np.sin(1.)])    # [px, py, theta]
xf0 = np.array([0.1, 8.5, np.cos(0.1), np.sin(0.1)])


# for cost
# leader scenario 1: near the follower
Ql1 = np.diag([2, 2, 0., 0.])
Ql2 = np.diag([1, 1, 0., 0.])
Rl = np.diag([2., 1])

# leader scenario 2: far from the follower
Ql1_s2 = np.diag([2, 2, 0., 0.])
Ql2_s2 = np.diag([.1, .1, 0., 0.])
Rl_s2 = np.diag([2., 1])

# follower's cost
Qf1 = np.diag([10., 10., 0, 0])
Qf2 = np.diag([.1, .1, 0, 0])
Qf3 = -1.
Rf = np.diag([2., 0.05])


# for nn_fdyn training
nn_fdyn = {}
nn_fdyn['linear1'] = [dimxf+dimxl+dimul, 30]
nn_fdyn['linear2'] = [30, 30]
nn_fdyn['linear3'] = [30, dimxf]

nn_fdyn['lr'] = 1e-4
nn_fdyn['mon'] = 0.8
nn_fdyn['batch'] = 32
nn_fdyn['epoch'] = 800


# for kp_nn training
kp_nn = {}
kp_nn['linear1'] = [dimxf, 40]
kp_nn['linear2'] = [40, 40]
kp_nn['linear3'] = [40, 10]

kp_nn['pred_horizon'] = 30
kp_nn['gam'] = 0.9
kp_nn['lr'] = 1e-4      # SGD learning rate
kp_nn['mom'] = 0.6      # SGD momentum
kp_nn['batch_size'] = 64
kp_nn['n_epoch'] = 1000

# for dmd
dmd = {}
dmd['gam'] = 0.9
dmd['pred_horizon'] = 30


def print_key_param():
    print('Parameters:')
    print('seed: ', seed)
    print('obs: ', obs_spec)
    print('xd: ', xd)
    print('dt: ', dt)
    print('dimT: ', dimT)
    print('xl0: ', xl0)
    print('xf0: ', xf0)
    print('xd: ', xd)

    print('Qf1: ', Qf1)
    print('Qf2: ', Qf2)
    print('Qf3: ', Qf3)
    print('Rf: ', Rf)

    print('Ql1: ', Ql1)
    print('Ql2: ', Ql2)
    print('Rl: ', Rl)
    print('Ql1_s2: ', Ql1_s2)
    print('Ql2_s2: ', Ql2_s2)

print_key_param()