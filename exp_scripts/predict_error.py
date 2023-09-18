'''
Predict error of K-step trajectory using learned Koopman operator.
'''
import sys, os
sys.path.insert(1, os.path.realpath('.'))

import numpy as np
import torch
from sg_koopman.common.utils import PlotUtilities


def prederr_fdynbr():
    from sg_koopman.nn_fdynbr import FdynTrain
    fdynbr_trainner = FdynTrain()
    pltutil = PlotUtilities()
    data = np.load('data/follower_fdynbr_lmix.npy') # or use testbench
    N = [2000, 4000, 6000, 8000]
    for i in range(len(N)):
        fnet = torch.load('data/nn_fdyn/exp1/model_N'+str(N[i])+'/fdynbr.pt')

        err_mean, err_std = fdynbr_trainner.predict_error(data, fnet, N=20, K=10)
        print(f'Model N_{N[i]}:')
        print('err_mean: ', err_mean)
        print('err_std: ', err_std)
        pltutil.plot_prederr(err_mean, err_std)
        a = 1


def prederr_dmd():
    from sg_koopman.dmd import DMDTrain
    dmd_trainner = DMDTrain()
    pltutil = PlotUtilities()
    data = np.load('data/follower_fdynbr_lmix.npy') # or use testbench
    N = [2000, 4000, 6000, 8000]
    for i in range(len(N)):
        Kp = np.load('data/dmd/model_N'+str(N[i])+'/Kp.npy')

        err_mean, err_std = dmd_trainner.predict_error(data, Kp, N=20, K=10)
        print(f'Model N_{N[i]}:')
        print('err_mean: ', err_mean)
        print('err_std: ', err_std)
        pltutil.plot_prederr(err_mean, err_std)
        a = 1


def prederr_kp_nn():
    from sg_koopman.kp_nn import KpTrain
    kp_trainner = KpTrain()
    pltutil = PlotUtilities()
    data = np.load('data/follower_fdynbr_lmix.npy') # or use testbench
    N = [2000, 4000, 6000, 8000]
    for i in range(len(N)):
        kpnetdict = torch.load('data/kp_nn/model_N'+str(N[i])+'/bases.pt')
        operator = torch.load('data/kp_nn/model_N'+str(N[i])+'/operators.pt')

        err_mean, err_std = kp_trainner.predict_error(data, kpnetdict, operator, N=20, K=10)
        print(f'Model N_{N[i]}:')
        print('err_mean: ', err_mean)
        print('err_std: ', err_std)
        pltutil.plot_prederr(err_mean, err_std)
        a = 1


if __name__ == '__main__':    
    #prederr_fdynbr()
    prederr_kp_nn()
    #prederr_dmd()