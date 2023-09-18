'''
Receding horizon planning for different algorithms.
'''
import sys, os
sys.path.insert(1, os.path.realpath('.'))

import numpy as np
import torch


def rhp_kp():
    from sg_koopman.kp_nn import rh_planning

    kpnetdict = torch.load('data/kp_nn/model_N8000/bases.pt')
    operator = torch.load('data/kp_nn/model_N8000/operators.pt')
    xl_traj, ul_traj, xf_traj, uf_traj = rh_planning(kpnetdict, operator)

    # save trajectory data
    np.save('data/kp_nn/xl_traj_rh.npy', xl_traj)
    np.save('data/kp_nn/xf_traj_rh.npy', xf_traj)
    np.save('data/kp_nn/ul_traj_rh.npy', ul_traj)
    np.save('data/kp_nn/uf_traj_rh.npy', uf_traj)


def rhp_fdynbr():
    from sg_koopman.nn_fdynbr import rh_planning

    fnetdict = torch.load('data/nn_fdynbr/model_N8000/fdynbr.pt')
    xl_traj, ul_traj, xf_traj, uf_traj = rh_planning(fnetdict)
    
    # save trajectory data
    np.save('data/nn_fdynbr/xl_traj_rc.npy', xl_traj)
    np.save('data/nn_fdynbr/xf_traj_rc.npy', xf_traj)
    np.save('data/nn_fdynbr/ul_traj_rc.npy', ul_traj)
    np.save('data/nn_fdynbr/uf_traj_rc.npy', uf_traj)


def rhp_dmd():
    from sg_koopman.dmd import rh_planning

    Kp = torch.load('data/dmd/model_N8000/Kp.npy')
    xl_traj, ul_traj, xf_traj, uf_traj = rh_planning(Kp)

    # save trajectory data
    np.save('data/dmd/xl_traj_rc.npy', xl_traj)
    np.save('data/dmd/xf_traj_rc.npy', xf_traj)
    np.save('data/dmd/ul_traj_rc.npy', ul_traj)
    np.save('data/dmd/uf_traj_rc.npy', uf_traj)


def rhp_nonlin():
    from sg_koopman.nonlin_ocp import rh_planning

    xl_traj, ul_traj, xf_traj, uf_traj = rh_planning()
    
    # save trajectory data
    np.save('data/nonlin_ocp/xl_traj_rc.npy', xl_traj)
    np.save('data/nonlin_ocp/xf_traj_rc.npy', xf_traj)
    np.save('data/nonlin_ocp/ul_traj_rc.npy', ul_traj)
    np.save('data/nonlin_ocp/uf_traj_rc.npy', uf_traj)
    


if __name__ == '__main__':
    rhp_kp()
    rhp_nonlin()
    rhp_fdynbr()
    rhp_dmd()