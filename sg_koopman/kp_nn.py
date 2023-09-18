"""
This module implements koopman learning for partial dynamics to perform receding horizon planning.
"""
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint
import torch
from sg_koopman.common.agents import Leader, Follower
from sg_koopman.common.utils import to_torch, to_numpy, PlotUtilities
from sg_koopman.common.models import KpNetPartial
import sg_koopman.parameters as param
from os import mkdir
from os.path import exists


class KpTrain:
    def __init__(self) -> None:
        self.dimxl, self.dimul = param.dimxl, param.dimul
        self.dimxf, self.dumuf = param.dimxf, param.dimuf

        # learning parameters
        self.pred_horizon = param.kp_nn['pred_horizon']
        self.gam = param.kp_nn['gam']
        self.lr = param.kp_nn['lr']
        self.mom = param.kp_nn['mom']
        self.batch_size = param.kp_nn['batch_size']
        self.n_epoch = param.kp_nn['n_epoch']


    def get_train_data(self, data, N=1000):
        '''
        Prepare interactive trajectory data. D[i, k, :] = [xf_k, uf*_k, xf_kp1, xl_k, ul_k, xl_kp1]
        '''
        from os.path import exists

        if data is None or len(data=0):
            if exists('data/follower_dynbr_lmix.npy'):
                raise Exception('No data provided or found.')
            else:
                data = np.load('data/follower_dynbr_lmix.npy')
        
        if N <= data.shape[0]:
            data = data[np.random.choice(data.shape[0], N, replace=False), :]
        else:
            raise Exception('N exceeds totoal number of data samples.')

        idx =  data.shape[0]*4//5   # 80% for training and 20% for testing 
        Dtrain, Dtest = data[: idx, :], data[idx: , :]
        print(f'Dtrain size: {Dtrain.shape[0]}, Dtest size {Dtest.shape[0]}.')
        return Dtrain, Dtest


    def get_batch_data(self, data, batch_size):
        '''
        This function generates batch data for training and testing. D[i, k, :] = [xf_k, uf_k, xf_kp1, xl_k, ul_k, xl_kp1]. 
        Useful data: [xf, xl, ul, xf_kp1]
        Each batch data = [X, U]. 
        - X = [xf]: batch_size x (K+1) x dimxf, U = [xl, ul]: batch_size x K x (dimxl+dimul).
        '''
        batch_data = []
        n_batch = data.shape[0] // batch_size
        for i in range(n_batch):
            idx = i * batch_size
            X = data[idx: idx+batch_size, :, 0: self.dimxf]     # [xf_k]
            XF = data[idx: idx+batch_size, [-1], self.dimxf+self.dimuf: 2*self.dimxf+self.dimuf]    # [xf_K]
            X = torch.cat((X, XF), dim=1)

            U = data[idx: idx+batch_size, :, 2*self.dimxf+self.dimuf: 2*self.dimxf+self.dimuf+self.dimxl+self.dimul]    # [xl_k, ul_k]
            batch_data.append([X, U])
        return batch_data


    def train_kp_nn(self, Dtrain, Dtest=None, device='cpu'):
        '''
        Train koopman operator and embedding.
        '''
        Dtrain = torch.tensor(Dtrain, device=device)
        Dtest = torch.tensor(Dtest, device=device)

        kpnet = KpNetPartial()
        kpnet.to(device)

        # Koopman operator matrices, y = [xf, kp(xf)], C @ y = xf
        # torch.manual_seed(seed)
        A = torch.rand(self.dimxf+kpnet.dimout, self.dimxf+kpnet.dimout, requires_grad=True, device=device)
        B = torch.rand(self.dimxf+kpnet.dimout, self.dimxl+self.dimul, requires_grad=True, device=device)
        C = torch.hstack((torch.eye(self.dimxf), torch.zeros(self.dimxf,kpnet.dimout))).to(device)

        loss_fn = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD([A, B] + list(kpnet.parameters()), lr=self.lr, momentum=self.mom)
        #optimizer = torch.optim.Adam([A, B] + list(kpnet.parameters()), lr=1e-4)
        train_loss = []

        if Dtest is not None:
            test_loss = np.zeros(self.n_epoch)
            Dtest_batch_data = self.get_batch_data(Dtest, Dtest.shape[0])   # only one-batch test data

        import time
        start_t = time.time()

        for t in range(self.n_epoch):
            print("Epoch: {}/{}\n-------------".format(t+1, self.n_epoch))
            Dtrain = Dtrain[torch.randperm(Dtrain.shape[0]), :]     # shuffle training data in each epoch
            batch_data = self.get_batch_data(Dtrain, self.batch_size)
            batch_train_loss = np.zeros(len(batch_data))

            for i in range(len(batch_data)):   # iterate over all training data organized by batch
                X, U = batch_data[i][0], batch_data[i][1]
            
                # K step prediction loss
                loss = 0
                for k in range(self.pred_horizon):
                    loss += loss_fn( A @ torch.hstack( (X[:,k,:],kpnet(X[:,k,:])) ).T + B @ U[:,k,:].T, 
                                    torch.hstack( (X[:,k+1,:],kpnet(X[:,k+1,:])) ).T ) * (self.gam**k) / self.batch_size
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_train_loss[i] = loss.item()

                if (i+1) % 10 == 0:
                    print('    iter: {}/{}, loss: {:.7f}'.format(i+1, len(batch_data), loss.item()))
            train_loss.append(batch_train_loss)

            # validation/test loss
            if Dtest is not None:
                X, U = Dtest_batch_data[0][0], Dtest_batch_data[0][1]
                loss = 0
                for k in range(self.pred_horizon):
                    loss += loss_fn( A @ torch.hstack( (X[:,k,:],kpnet(X[:,k,:])) ).T + B @ U[:,k,:].T, 
                                    torch.hstack( (X[:,k+1,:],kpnet(X[:,k+1,:])) ).T ) * (self.gam**k) / Dtest.shape[0]
                test_loss[t] = loss.item()

        end_t = time.time()
        print("Elapsed time: {:.3f} min.".format((end_t-start_t)/60))
    
        kp_op = {'A': A.detach().cpu(), 'B': B.detach().cpu(), 'C': C}
        tl = np.mean(np.vstack(train_loss), axis=1)

        # save the follower's trained dynamical model
        save_flag = True
        if save_flag:
            model_dir_name = 'data/kp_nn/model_N' + str(Dtrain.shape[0]) + '/'
            if not exists(model_dir_name):
                mkdir(model_dir_name)
            
            torch.save(kp_op, model_dir_name+'operators.pt')
            torch.save(kpnet.state_dict(), model_dir_name+'bases.pt')
            np.save(model_dir_name+'train_loss.npy', tl)
            np.save(model_dir_name+'test_loss.npy', test_loss)
    
        return kpnet.state_dict(), kp_op


    def train_kp_nn_embeddingonly(self, Dtrain, Dtest=None, device='cpu'):
        '''
        Train embedding only. Use pinv to estimate koopman operator.
        '''
        Dtrain = torch.tensor(Dtrain, device=device)
        Dtest = torch.tensor(Dtest, device=device)

        #l = Leader()
        #f = Follower()
        kpnet = KpNetPartial()
        kpnet.to(device)
        C = torch.hstack((torch.eye(self.dimxf), torch.zeros(self.dimxf,kpnet.dimout))).to(device)        # C @ y = xf

        loss_fn = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(kpnet.parameters(), lr=self.lr, momentum=self.mom)
        #optimizer = torch.optim.Adam(kpnet.parameters(), lr=1e-4)
        train_loss = []

        if Dtest is not None:
            test_loss = np.zeros(self.n_epoch)
            Dtest_batch_data = self.get_batch_data(Dtest, Dtest.shape[0])  # only one-batch test data

        import time
        start_t = time.time()

        for t in range(self.n_epoch):
            print("Epoch: {}/{}\n-------------".format(t+1, self.n_epoch))
            Dtrain = Dtrain[torch.randperm(Dtrain.shape[0]), :]     # shuffle training data in each epoch
            batch_data = self.get_batch_data(Dtrain, self.batch_size)
            batch_train_loss = np.zeros(len(batch_data))

            for i in range(len(batch_data)):   # iterate over all training data organized by batch
                X, U = batch_data[i][0], batch_data[i][1]

                # compute Koopman operator matrices using pinv
                G = torch.zeros(self.dimxf+kpnet.dimout+self.dimxl+self.dimul, self.dimxf+kpnet.dimout+self.dimxl+self.dimul)
                P = torch.zeros(self.dimxf+kpnet.dimout, self.dimxf+kpnet.dimout+self.dimxl+self.dimul)
                for k in range(self.pred_horizon):
                    Ykp1 = torch.hstack( (X[:,k+1,:], kpnet(X[:,k+1,:])) )
                    Wk = torch.hstack( (X[:,k,:],kpnet(X[:,k,:]), U[:,k,:]) )
                    G += Wk.T @ Wk / self.batch_size
                    P += Ykp1.T @ Wk / self.batch_size
                K = P @ torch.linalg.pinv(G)
                A = K[:, : self.dimxf+kpnet.dimout].detach()
                B = K[:, self.dimxf+kpnet.dimout: ].detach()
            
                # K step prediction loss
                loss = 0
                for k in range(self.pred_horizon):
                    loss += loss_fn( A @ torch.hstack( (X[:,k,:],kpnet(X[:,k,:])) ).T + B @ U[:,k,:].T, 
                                    torch.hstack( (X[:,k+1,:],kpnet(X[:,k+1,:])) ).T ) * (self.gam**k) / self.batch_size
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_train_loss[i] = loss.item()

                if (i+1) % 10 == 0:
                    print('    iter: {}/{}, loss: {:.7f}'.format(i+1, len(batch_data), loss.item()))
            train_loss.append(batch_train_loss)

            # validation/test loss
            if Dtest is not None:
                X, U = Dtest_batch_data[0][0], Dtest_batch_data[0][1]
            
                # compute Koopman operator matrices using pinv
                G = torch.zeros(self.dimxf+kpnet.dimout+self.dimxl+self.dimul, self.dimxf+kpnet.dimout+self.dimxl+self.dimul)
                P = torch.zeros(self.dimxf+kpnet.dimout, self.dimxf+kpnet.dimout+self.dimxl+self.dimul)
                for k in range(self.pred_horizon):
                    Ykp1 = torch.hstack( (X[:,k+1,:], kpnet(X[:,k+1,:])) )
                    Wk = torch.hstack( (X[:,k,:],kpnet(X[:,k,:]), U[:,k,:]) )
                    G += Wk.T @ Wk / Dtest.shape[0]
                    P += Ykp1.T @ Wk / Dtest.shape[0]
                K = P @ torch.linalg.pinv(G)
                A = K[:, : self.dimxf+kpnet.dimout].detach()
                B = K[:, self.dimxf+kpnet.dimout: ].detach()

                loss = 0
                for k in range(self.pred_horizon):
                    loss += loss_fn( A @ torch.hstack( (X[:,k,:],kpnet(X[:,k,:])) ).T + B @ U[:,k,:].T, 
                                    torch.hstack( (X[:,k+1,:],kpnet(X[:,k+1,:])) ).T ) * (self.gam**k) / Dtest.shape[0]
                test_loss[t] = loss.item()

        end_t = time.time()
        print("Elapsed time: {:.3f} min.".format((end_t-start_t)/60))
    
        kp_op = {'A': A.detach().cpu(), 'B': B.detach().cpu(), 'C': C}
        tl = np.mean(np.vstack(train_loss), axis=1)

        # save the follower's trained dynamical model
        save_flag = True
        if save_flag:
            model_dir_name = 'data/kp_nn/model_embed_N' + str(Dtrain.shape[0]) + '/'
            if not exists(model_dir_name):
                mkdir(model_dir_name)
            
            torch.save(kp_op, model_dir_name+'operators.pt')
            torch.save(kpnet.state_dict(), model_dir_name+'bases.pt')
            np.save(model_dir_name+'train_loss.npy', tl)
            np.save(model_dir_name+'test_loss.npy', test_loss)
    
        return kpnet.state_dict(), kp_op


    def predict_error(self, data, kpnetdict, operator, N=20, K=5):
        '''
        Compute K-step prediction error (mean and std) using N ground truth data trajectories.
        '''
        data = data[self.rng.choice(data.shape[0], N, replace=False), :]
        data = torch.tensor(data)

        kpnet = KpNetPartial()
        kpnet.load_state_dict(kpnetdict)
        
        error = torch.zeros((N, K))
        for i in range(N):
            xl_traj = data[:, 2*self.dimxf+self.dimuf: 2*self.dimxf+self.dimuf+self.dimxl]  # [xl_0, ..., xl_Km1]
            ul_traj = data[:, 2*self.dimxf+self.dimuf+self.dimxl: 2*self.dimxf+self.dimuf+self.dimxl+self.dimul]    # [ul_0, ..., ul_Km1]
            xf_pred = torch.zeros((K, self.dimxf))  # does not contain xf_0
            xf_k = data[i, 0, :self.dimxf]   # xf_0
            y_k = torch.cat( (xf_k, kpnet(xf_k)) )  # y_0
            for k in range(K):
                # y_kp1 = A @ y_k + B @ [xl_k, ul_k]
                y_kp1 = operator['A'] @ y_k + operator['B'] @ torch.cat((xl_traj[k],ul_traj[k]))
                xf_pred[k, :] = operator['C'] @ y_kp1
                y_k = y_kp1
            
            xf_traj = data[i, :, self.dimxf+self.dimuf: 2*self.dimxf+self.dimuf]  # [xf_1, xf_2, ..., xf_K]
            error[i, :] = torch.linalg.norm(xf_pred-xf_traj, dim=1)
        err_mean = torch.mean(error, dim=0)
        err_std = torch.std(error, dim=0)
        return err_mean.detach().numpy(), err_std.detach().numpy()


def rh_planning(kpnetdict, operator):
    kpnet = KpNetPartial()
    kpnet.load_state_dict(kpnetdict)
    
    leader = Leader()
    follower = Follower()
    ocp_solver = OptimalControl()
    pltutil = PlotUtilities()
    
    xl_traj, ul_traj = [], [] 
    xf_traj, uf_traj = [], []

    xl, xf = leader.xl0, follower.xf0
    xl_traj.append(xl)
    xf_traj.append(xf)

    RC_MAX = 100    # max receding horizon
    X_pre = None
    import time
    for t in range(RC_MAX):
        print(f'------- RH planning iteration: {t+1} -------')
        st = time.time()
        ul, X_pre = ocp_solver.ocp(xl, xf, kpnet, operator, X_pre)
        et = time.time()
        print(f'  planning time: {(et-st):.4f}s')

        xl_new = leader.dyn(xl, ul)
        uf = follower.get_br(xf, xl_new)
        xf_new = follower.dyn(xf, uf)

        ul_traj.append(ul)
        uf_traj.append(uf)
        xl_traj.append(xl_new)
        xf_traj.append(xf_new)
        xl = xl_new
        xf = xf_new

        pltutil.plot_traj(np.array(xl_traj), np.array(xf_traj))
        # stop condition: t > RC_MAX or near target
    
    xl_traj, ul_traj = np.array(xl_traj), np.array(ul_traj)
    xf_traj, uf_traj = np.array(xf_traj), np.array(uf_traj)

    return xl_traj, ul_traj, xf_traj, uf_traj


class OptimalControl(Leader):
    '''
    Leader's optimal control problem. X = [xl1,xl2,..., y1,y2,..., ul1,ul2...]
    '''
    def __init__(self) -> None:
        super().__init__()
    

    def ocp(self, xl0, xf0, kpnet, operator, X_pre=None):
        '''
        Optimal control with Koopman model. X = [xl1,xl2,..., y1,y2,..., ul1,ul2...]
        '''
        A, B, C = to_numpy(operator['A']), to_numpy(operator['B']), to_numpy(operator['C'])
        B1, B2 = B[:, : self.dimxl], B[:, self.dimxl: ]
        dimy = A.shape[0]
        constr = []
        y0 = np.concatenate( (xf0, to_numpy(kpnet(to_torch(xf0)))) )        # construct y0
        yd = np.concatenate( (self.xd, to_numpy(kpnet(to_torch(self.xd)))) )    # construct yd

        # follower's linear dynamics: y_kp1 = A @ y_k + B1 @ xl_k + B2 @ ul_k
        tmp = np.diag(np.ones(self.dimT-1), k=-1)
        aa1 = np.kron(np.eye(self.dimT), -np.eye(dimy))
        aa2 = np.kron(tmp, A)
        AA = aa1 + aa2
        BB1 = np.kron(tmp, B1)
        BB2 = np.kron(np.eye(self.dimT), B2)
        bigA = np.hstack( (BB1, AA, BB2) )
        b = np.zeros(self.dimT*dimy)
        b[: dimy] = - A @ y0 - B1 @ xl0
        constr.append(LinearConstraint(bigA, lb=b, ub=b))

        def pred_dyn_tmp(X):
            xf_traj = np.zeros((self.dimT+1, self.dimxf))
            xf_traj[0,:] = xf0
            for t in range(self.dimT):
                i_y = np.arange(t*dimy, (t+1)*dimy) + self.dimT*self.dimxl
                y_tp1 = X[i_y]
                xf_traj[t+1, :] = C @ y_tp1
        
            xf_pred = np.zeros((self.dimT+1, self.dimxf))
            xf_pred[0,:] = xf0
            y_k = np.concatenate( (xf0, to_numpy(kpnet(to_torch(xf0)))) )
            for t in range(self.dimT):
                i_xl = np.arange(t*self.dimxl, (t+1)*self.dimxl)
                i_ul = np.arange(t*self.dimul, (t+1)*self.dimul) + self.dimT*(self.dimxl+dimy)
                if t == 0:
                    xl_k = xl0
                else:
                    xl_k = X[i_xl-self.dimxl]
                ul_k = X[i_ul]
                y_kp1 = A @ y_k + B1 @ xl_k + B2 @ ul_k
                xf_pred[t+1:, ] = C @ y_kp1
                y_k = y_kp1
            print(xf_traj)
            print(xf_pred)
            return xf_traj, xf_pred
    
        def myobj(X):
            J = 0
            for t in range(self.dimT):
                i_xl = np.arange(t*self.dimxl, (t+1)*self.dimxl)
                i_y = np.arange(t*dimy, (t+1)*dimy) + self.dimT*self.dimxl
                i_ul = np.arange(t*self.dimul, (t+1)*self.dimul) + self.dimT*(self.dimxl+dimy)
                xl_tp1, y_tp1, ul_t = X[i_xl], X[i_y], X[i_ul]
                J += (xl_tp1 - C@y_tp1) @ self.Ql1 @ (xl_tp1 - C@y_tp1) + (xl_tp1 - self.xd) @ self.Ql2 @ (xl_tp1 - self.xd) + ul_t @ self.Rl @ ul_t
            return J
        def jac_myobj(X):
            dJ = np.zeros_like(X)
            for t in range(self.dimT):
                i_xl = np.arange(t*self.dimxl, (t+1)*self.dimxl)
                i_y = np.arange(t*dimy, (t+1)*dimy) + self.dimT*self.dimxl
                i_ul = np.arange(t*self.dimul, (t+1)*self.dimul) + self.dimT*(self.dimxl+dimy)
                xl_tp1, y_tp1, ul_t = X[i_xl], X[i_y], X[i_ul]
            
                dJ[i_xl] = 2*self.Ql1 @ (xl_tp1 - C@y_tp1) + 2*self.Ql2 @ (xl_tp1 - self.xd)
                dJ[i_y] = -2*C.T @ self.Ql1 @ (xl_tp1 - C@y_tp1)
                dJ[i_ul] = 2*self.Rl @ ul_t
            return dJ
        def myobj_s2(X):
            J = 0
            for t in range(self.dimT):
                i_xl = np.arange(t*self.dimxl, (t+1)*self.dimxl)
                i_y = np.arange(t*dimy, (t+1)*dimy) + self.dimT*self.dimxl
                i_ul = np.arange(t*self.dimul, (t+1)*self.dimul) + self.dimT*(self.dimxl+dimy)
                xl_tp1, y_tp1, ul_t = X[i_xl], X[i_y], X[i_ul]
                J += (xl_tp1 - C@y_tp1) @ self.Ql1_s2 @ (xl_tp1 - C@y_tp1) + (xl_tp1 - self.xd) @ self.Ql2_s2 @ (xl_tp1 - self.xd) + ul_t @ self.Rl_s2 @ ul_t
            return J
        def jac_myobj_s2(X):
            dJ = np.zeros_like(X)
            for t in range(self.dimT):
                i_xl = np.arange(t*self.dimxl, (t+1)*self.dimxl)
                i_y = np.arange(t*dimy, (t+1)*dimy) + self.dimT*self.dimxl
                i_ul = np.arange(t*self.dimul, (t+1)*self.dimul) + self.dimT*(self.dimxl+dimy)
                xl_tp1, y_tp1, ul_t = X[i_xl], X[i_y], X[i_ul]
            
                dJ[i_xl] = 2*self.Ql1_s2 @ (xl_tp1 - C@y_tp1) + 2*self.Ql2_s2 @ (xl_tp1 - self.xd)
                dJ[i_y] = -2*C.T @ self.Ql1_s2 @ (xl_tp1 - C@y_tp1)
                dJ[i_ul] = 2*self.Rl_s2 @ ul_t
            return dJ

        def con_dyn(X):     # leader's dynamics only xl_tp1 = fL(xl_t, ul_t)
            f = np.zeros(self.dimT*self.dimxl)
            for t in range(self.dimT):
                i_xl = np.arange(t*self.dimxl, (t+1)*self.dimxl)
                i_ul = np.arange(t*self.dimul, (t+1)*self.dimul) + self.dimT*(self.dimxl+dimy)
                xl_tp1, ul_t = X[i_xl], X[i_ul]
                if t == 0:
                    xl_t = xl0
                else:
                    xl_t = X[i_xl-self.dimxl]
            
                f[i_xl] = xl_tp1 - self.dyn(xl_t, ul_t)
            return f
        def jac_con_dyn(X):
            df = np.zeros((self.dimT*self.dimxl, X.shape[0]))
            for t in range(self.dimT):
                i_xl = np.arange(t*self.dimxl, (t+1)*self.dimxl)
                i_ul = np.arange(t*self.dimul, (t+1)*self.dimul) + self.dimT*(self.dimxl+dimy)
                xl_tp1, ul_t = X[i_xl], X[i_ul]
                if t == 0:
                    xl_t = xl0
                else:
                    xl_t = X[i_xl-self.dimxl]
            
                dfl_dx, dfl_du = self.dyn_jac(xl_t, ul_t)
                df[t*self.dimxl: (t+1)*self.dimxl, i_xl] = np.eye(self.dimxl)
                df[t*self.dimxl: (t+1)*self.dimxl, i_ul] = -dfl_du
                if t > 0:
                    df[t*self.dimxl: (t+1)*self.dimxl, i_xl-self.dimxl] = -dfl_dx
            return df
        lb_dyn = np.zeros(self.dimT*self.dimxl)
        constr.append( NonlinearConstraint(con_dyn, lb=lb_dyn, ub=lb_dyn, jac=jac_con_dyn) )

        def con_safe(X):    # leader's safety constraint only: c_i(xl_t) >= 0
            f = np.zeros(self.dimT*len(self.obs))
            for t in range(self.dimT):
                i_xl = np.arange(t*self.dimxl, (t+1)*self.dimxl)
                f[t*len(self.obs): (t+1)*len(self.obs)] = self.safety_constr(X[i_xl])
            return f
        def jac_con_safe(X):
            df = np.zeros((self.dimT*len(self.obs), X.shape[0]))
            for t in range(self.dimT):
                i_xl = np.arange(t*self.dimxl, (t+1)*self.dimxl)
                df[t*len(self.obs): (t+1)*len(self.obs), i_xl] = self.safety_constr_jac(X[i_xl])
            return df
        lb_safe = np.ones(self.dimT*len(self.obs))
        constr.append( NonlinearConstraint(con_safe, 0.1*lb_safe, np.inf*lb_safe, jac=jac_con_safe) )

        # leader's input constraints
        BB = np.kron(np.eye(self.dimT), np.zeros((self.dimul,self.dimxl+self.dimul+dimy)))
        BB[:, self.dimT*(self.dimxl+dimy): ] = np.kron(np.eye(self.dimT), np.eye(self.dimul))
        lb_a = np.kron(np.ones(self.dimT), np.array([self.vmin, self.wmin]))
        ub_a = np.kron(np.ones(self.dimT), np.array([self.vmax, self.wmax]))
        constr.append(LinearConstraint(BB, lb_a, ub_a))

        '''
        #---------- test jac -----------#
        from scipy.optimize import approx_fprime, check_grad
        tmp = np.zeros(10)
        for ii in range(tmp.shape[0]):
            X00 = np.random.rand(self.dimT*(self.dimxl+dimy+self.dimul)) * 5 - 1
            #tmp[ii] = check_grad(myobj, jac_myobj, X00)
            #tmp[ii] = check_grad(myobj_s2, jac_myobj_s2, X00)
            #tmp[ii] = check_grad(con_dyn, jac_con_dyn, X00)
            tmp[ii] = check_grad(con_safe, jac_con_safe, X00)
        print(tmp)
        #-------------------------------#
        '''

        # scenario selection 
        sce_th = 1
        if np.linalg.norm(xl0[:2]-xf0[:2]) <= sce_th or X_pre is None:   # scenario 1, near the follower, guide to target
            X0 = self.init_1(xl0, xf0, kpnet)
            res = minimize(myobj, X0, jac=jac_myobj, constraints=constr, options={'maxiter': 200, 'disp': False})
            print('  scenario 1: status {}: {}'.format(res.status, res.message))
        else:   # scenario 2, far from the follower, go to the follower
            X0 = self.init_2(xl0, xf0, kpnet)
            #X0 = X_pre
            res = minimize(myobj_s2, X0, jac=jac_myobj_s2, constraints=constr, options={'maxiter': 200, 'disp': False})
            print('  scenario 2: status {}: {}'.format(res.status, res.message))

        # convert results to trajectory
        ul_opt = res.x[self.dimT*(self.dimxl+dimy): ]
        ul_opt = ul_opt.reshape((self.dimT, self.dimul))
        print(f'  ulopt: {ul_opt[0, :]}')
    
        #pred_dyn_tmp(res.x)

        return ul_opt[0, :], res.x #, res.x[self.dimT*self.dimxl: self.dimT*self.dimxl+dimy]
    

    def init_1(self, xl0, xf0, kpnet):
        '''
        Initialize trajectory using Astar algorithm. Leader goes to the destination.
        '''
        dxdy = param.dxdy
        grid_map = param.grid_map

        # planning trajectory with Astar
        xl_traj = self.astar_x_traj(xl0, self.xd, grid_map, dxdy, horizon=self.dimT+1)    # xl_traj[0,:] = xl0
        xf_traj = self.astar_x_traj(xf0, xl0, grid_map, dxdy, horizon=self.dimT+1)

        # convert xf_traj to lifted space.
        dimy = self.dimxf+kpnet.dimout
        y_traj = np.zeros((xf_traj.shape[0], dimy))
        for i in range(y_traj.shape[0]):
            y_traj[i, :] = np.concatenate( (xf_traj[i, :], to_numpy(kpnet(to_torch(xf_traj[i, :])))) )

        # construct X0
        X0 = np.zeros(self.dimT*(self.dimxl+self.dimul+dimy))   # X0 = [xl, y, ul]
        tmp = np.concatenate( (xl_traj[1:,:].flatten(), y_traj[1:,:].flatten()) )
        X0[: tmp.size] = tmp
        X0[tmp.size:] = 0*np.random.rand(self.dimul*self.dimT)

        return X0


    def init_2(self, xl0, xf0, kpnet):
        '''
        Initialize trajectory using Astar algorithm. Leader goes to the follower.
        '''
        dxdy = param.dxdy
        grid_map = param.grid_map

        # planning trajectory with Astar
        xl_traj = self.astar_x_traj(xl0, xf0, grid_map, dxdy, horizon=self.dimT+1)
        xf_traj = self.astar_x_traj(xf0, xl0, grid_map, dxdy, horizon=self.dimT+1)

        # convert xf_traj to lifted space.
        dimy = self.dimxf+kpnet.dimout
        y_traj = np.zeros((xf_traj.shape[0], dimy))
        for i in range(y_traj.shape[0]):
            y_traj[i, :] = np.concatenate( (xf_traj[i, :], to_numpy(kpnet(to_torch(xf_traj[i, :])))) )
    
        # construct X0
        X0 = np.zeros(self.dimT*(self.dimxl+self.dimul+dimy))   # X0 = [xl, y, ul]
        tmp = np.concatenate( (xl_traj[1:,:].flatten(), y_traj[1:,:].flatten()) )
        X0[: tmp.size] = tmp
        X0[tmp.size:] = 0*np.random.rand(self.dimul*self.dimT)

        return X0
    

    def init_shiftX(self, X_pre, yd):
        '''
        This function initializes X given the previous solution of OCP.
        Set X[-1] = xd/yd and U[-1] = 0. The rest shift left.
        '''
        X0 = np.zeros_like(X_pre)
        dimy = yd.shape[0]
    
        i_xl = np.arange((self.dimT-1)*self.dimxl)
        X0[i_xl] = X_pre[i_xl+self.dimxl]
        X0[i_xl[-1]+1: i_xl[-1]+1+self.dimxl] = self.xd

        i_y = np.arange((self.dimT-1)*dimy) + self.dimT*self.dimxl
        X0[i_y] = X_pre[i_y+dimy]
        X0[i_y[-1]+1: i_y[-1]+1+dimy] = yd
    
        i_ul = np.arange((self.dimT-1)*self.dimul) + self.dimT*(self.dimxl+dimy)
        X0[i_ul] = X_pre[i_ul+self.dimul]
    
        return X0

