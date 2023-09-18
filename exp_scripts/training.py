'''
Training different algorithms.
'''
import sys, os
sys.path.insert(1, os.path.realpath('.'))

from sg_koopman.common.utils import PlotUtilities
import numpy as np

#DtrainN = [x*4//5 for x in N]

def train_nn_fdynbr():
    N = [2500, 5000, 7500, 10000]      # Dtrain=[2000, 4000, 6000, 8000]
    device = 'cpu'
    from sg_koopman.nn_fdynbr import FdynTrain
    pltutil = PlotUtilities()
    fdynbr_trainner = FdynTrain()
    data = np.load('data/follower_fdynbr_lmix.npy')
    for i in range(len(N)):
        print(f'Training nn_fdynbr with {N[i]} data and {device} ...')
        Dtrain, Dtest = fdynbr_trainner.get_train_data(data, N=N[i])
        fnet = fdynbr_trainner.train_fdynbr(Dtrain, Dtest, device)

        #fname = 'data/nn_fdynbr/model_N' + str(Dtrain.shape[0])
        #train_loss = np.load(fname+'/train_loss.npy')
        #test_loss = np.load(fname+'/test_loss.npy')
        #pltutil.plot_loss(train_loss[100:], test_loss[100:])
        a = 1


def train_dmd():
    N = [2500, 5000, 7500, 10000]      # Dtrain=[2000, 4000, 6000, 8000]
    from sg_koopman.dmd import DMDTrain
    pltutil = PlotUtilities()
    dmd_trainner = DMDTrain()
    data = np.load('data/follower_fdynbr_lmix.npy')
    for i in range(len(N)):
        print(f'Training dmd with {N[i]} data ...')
        Kp = dmd_trainner.find_dmd_kp(data, N=N[i])

    
def train_kp_nn():
    N = [2500, 5000, 7500, 10000]      # Dtrain=[2000, 4000, 6000, 8000]
    device = 'cuda:1'
    from sg_koopman.kp_nn import KpTrain
    pltutil = PlotUtilities()
    kp_trainner = KpTrain()
    data = np.load('data/follower_fdynbr_lmix.npy')
    for i in range(len(N)):
        print(f'Training kp_net with {N[i]} data and {device} ...')
        Dtrain, Dtest = kp_trainner.get_train_data(data, N=N[i])
        kpnet, kp_op = kp_trainner.train_kp_nn(Dtrain, Dtest, device)
        #kpnet, kp_op = kp_trainner.train_kp_nn_embeddingonly(Dtrain, Dtest, device)

        fname = 'data/kp_nn/model_N' + str(Dtrain.shape[0])
        #fname = 'data/kp_nn/model_embed_N' + str(Dtrain.shape[0])
        #train_loss = np.load(fname+'/train_loss.npy')
        #test_loss = np.load(fname+'/test_loss.npy')
        #pltutil.plot_loss(train_loss[400:], test_loss[400:])
        a = 1


def train_kp_nn_multiproc():
    import multiprocessing, time
    from sg_koopman.kp_nn import KpTrain
    N = [2500, 5000, 7500, 10000]
    kp_trainner = KpTrain()
    data = np.load('data/follower_fdynbr_lmix.npy')

    arg_list = []
    device_list = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
    for i in range(len(N)):     # parallel training
        device = device_list[i]
        print(f'Training kp_full with {N[i]} data and {device} ...')
        Dtrain, Dtest = kp_trainner.get_train_data(data, N=N[i])
        arg_list.append( (Dtrain, Dtest, device) )
    
    with multiprocessing.Pool() as pool:
        st = time.time()
        res_mp = pool.starmap(kp_trainner.train_kp_nn, arg_list)
        elapsed_time = time.time() - st
    print("total time for multi-processing: {:.3f} min".format(elapsed_time/60))



if __name__ == '__main__':
    #train_nn_fdynbr()
    #train_dmd()
    train_kp_nn()

    #train_kp_nn_multiproc()