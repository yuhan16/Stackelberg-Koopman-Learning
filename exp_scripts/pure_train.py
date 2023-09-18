import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint
import torch
from sg_koopman.common.agents import Leader, Follower
from sg_koopman.common.utils import add_method, to_torch, to_numpy, Utilities, SamplingUtilities
from sg_koopman.common.models import KpNetPartial
import sg_koopman.parameters as param
from os import mkdir
from os.path import exists


def train_kp_partial_dynonly(Dtrain, Dtest=None, device='cpu'):
    Dtrain = torch.tensor(Dtrain, device=device)
    Dtest = torch.tensor(Dtest, device=device)

    util = Utilities()
    l = Leader()
    f = Follower()
    kpnet = KpNetPartial()
    kpnet.to(device)

    # Koopman operator matrices
    # torch.manual_seed(seed)
    A = torch.rand(f.dimxf+kpnet.dimout, f.dimxf+kpnet.dimout, requires_grad=True, device=device)
    B = torch.rand(f.dimxf+kpnet.dimout, f.dimuf, requires_grad=True, device=device)
    C = torch.hstack((torch.eye(f.dimxf), torch.zeros(f.dimxf,kpnet.dimout))).to(device)        # C @ y = xf

    # learning parameters
    pred_horizon = min(15, Dtrain.shape[1])    # or use K
    gam = 0.9
    lr = 1e-4
    mom = 0.6
    batch_size = 64  #32
    n_epoch = 300

    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD([A, B] + list(kpnet.parameters()), lr=lr, momentum=mom)
    #optimizer = torch.optim.Adam([A, B] + list(kpnet.parameters()), lr=1e-4)
    train_loss = []

    if Dtest is not None:
        test_loss = np.zeros(n_epoch)
        Dtest_batch_data = get_batch_data_dynonly(Dtest, Dtest.shape[0])  # only one-batch test data

    import time
    start_t = time.time()

    for t in range(n_epoch):
        print("Epoch: {}/{}\n-------------".format(t+1, n_epoch))
        #util.rng.shuffle(Dtrain)   # shuffle training data in each epoch
        Dtrain = Dtrain[torch.randperm(Dtrain.shape[0]), :]
        batch_data = get_batch_data_dynonly(Dtrain, batch_size)
        batch_train_loss = np.zeros(len(batch_data))

        for i in range(len(batch_data)):   # iterate over all training data organized by batch
            X, U = batch_data[i][0], batch_data[i][1]
            
            # K step prediction loss
            loss = 0
            for k in range(pred_horizon):
                loss += loss_fn( A @ torch.hstack( (X[:,k,:],kpnet(X[:,k,:])) ).T + B @ U[:,k,:].T, 
                                torch.hstack( (X[:,k+1,:],kpnet(X[:,k+1,:])) ).T ) * (gam**k) / batch_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_train_loss[i] = loss.item()

            if (i+1) % 10 == 0:
                print('    iter: {}/{}, loss: {:.7f}'.format(i+1, len(batch_data), loss.item()))    # loss.item()/batch_size
        train_loss.append(batch_train_loss)

        # validation/test loss
        if Dtest is not None:
            X, U = Dtest_batch_data[0][0], Dtest_batch_data[0][1]
            loss = 0
            for k in range(pred_horizon):
                loss += loss_fn( A @ torch.hstack( (X[:,k,:],kpnet(X[:,k,:])) ).T + B @ U[:,k,:].T, 
                                torch.hstack( (X[:,k+1,:],kpnet(X[:,k+1,:])) ).T ) * (gam**k) / Dtest.shape[0]
            test_loss[t] = loss.item()

    end_t = time.time()
    print("Elapsed time: {:.3f} min.".format((end_t-start_t)/60))
    
    # save the follower's trained dynamical model
    model_dir_name = 'model_dynonly_N' + str(Dtrain.shape[0]) + '/'
    if not exists(model_dir_name):
        mkdir(model_dir_name)
    
    Af, Al, Bl, b = train_linear_br(Dtrain) # find linear approximation of br function

    kp_op = {'A': A.detach(), 'B': B.detach(), 'C': C, 'Af': Af, 'Al': Al, 'Bl': Bl, 'b': b}
    torch.save(kp_op, model_dir_name+'operators.pt')
    torch.save(kpnet.state_dict(), model_dir_name+'bases.pt')

    tl = np.mean(np.vstack(train_loss), axis=1)
    np.save(model_dir_name+'train_loss.npy', tl)
    np.save(model_dir_name+'test_loss.npy', test_loss)
    
    return kpnet.state_dict(), kp_op

    
def train_linear_br(data):
    '''
    Use linear regression to learn the function: uf_opt = Af @ xf + Al @ xl + Bl @ ul + b.
    D[i,k,:] = [xf_k, uf_k, xf_kp1, xl_k, ul_k, xl_kp1]
    '''
    l = Leader()
    f = Follower()
    N, K = data.shape[0], data.shape[1]
    xf = data[:, :, 0: f.dimxf]
    uf = data[:, :, f.dimxf: f.dimxf+f.dimuf]
    xl = data[:, :, 2*f.dimxf+f.dimuf: 2*f.dimxf+f.dimuf+l.dimxl]
    ul = data[:, :, 2*f.dimxf+f.dimuf+l.dimxl: 2*f.dimxf+f.dimuf+l.dimxl+l.dimul]
    
    X = torch.cat((xf, xl, ul, torch.ones((N,K,1))), dim=2)  # augment element 1 for bias
    uf = uf.reshape((N*K, f.dimuf))
    X = X.reshape((N*K, f.dimxf+l.dimxl+l.dimul+1))
    
    G = X.T @ X / X.shape[0]
    P = uf.T @ X / X.shape[0]
    Kp = P @ torch.linalg.pinv(G)
    
    Af = Kp[:, :f.dimxf]
    Al = Kp[:, f.dimxf: f.dimxf+l.dimxl]
    Bl = Kp[:, f.dimxf+l.dimxl: f.dimxf+l.dimxl+l.dimul]
    b = Kp[:, -1]

    '''
    Dtest = data[:-1, :]
    xft = Dtest[:, :, 0: f.dimxf]
    uft = Dtest[:, :, f.dimxf: f.dimxf+f.dimuf]
    xlt = Dtest[:, :, 2*f.dimxf+f.dimuf: 2*f.dimxf+f.dimuf+l.dimxl]
    ult = Dtest[:, :, 2*f.dimxf+f.dimuf+l.dimxl: 2*f.dimxf+f.dimuf+l.dimxl+l.dimul]
    Xt = np.concatenate((xft, xlt, ult, np.ones((Dtest.shape[0],K,1))), axis=2)  # augment element 1 for bias
    uft = uft.reshape((Dtest.shape[0]*K, f.dimuf))
    Xt = Xt.reshape((Dtest.shape[0]*K, f.dimxf+l.dimxl+l.dimul+1))
    err = uft - (Xt @ Kp.T)
    a = np.linalg.norm(err, axis=1)

    import matplotlib.pyplot as plt
    _ = plt.hist(a, bins='auto')  # arguments are passed to np.histogram
    plt.title("Histogram with 'auto' bins")
    plt.savefig('tmp/tmp6.png')
    '''
    return Af, Al, Bl, b


def get_batch_data_dynonly(data, batch_size):
    '''
    This function generates batch data for training and testing. D[i, k, :] = [xf_k, uf_k, xf_kp1, xl_k, ul_k, xl_kp1]. 
    Useful data: [xf, uf, xf_kp1]
    Each batch data = [X, U]. 
      - X = [xf]: batch_size x (K+1) x dimxf, U = [uf]: batch_size x K x dimuf.
    '''
    l = Leader()
    f = Follower()
    batch_data = []
    n_batch = data.shape[0] // batch_size
    for i in range(n_batch):
        idx = i * batch_size
        X = data[idx: idx+batch_size, :, 0: f.dimxf]    # [xf_k]
        XF = data[idx: idx+batch_size, [-1], f.dimxf+f.dimuf: 2*f.dimxf+f.dimuf]  # [xf_K]
        X = torch.cat((X, XF), dim=1)
        U = data[idx: idx+batch_size, :, f.dimxf: f.dimxf+f.dimuf]  # [uf_k]
        batch_data.append([X, U])
    return batch_data
    


if __name__ == '__main__':
    Dtrain, Dtest = get_data_kp_partial(N=5000)
    train_linear_br(Dtest)
    #train_kp_partial_dynonly(Dtrain, Dtest)