"""
This module implements learning follower's dynamics with NN and perform receding horizon planning.
"""
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint
import torch
from sg_koopman.common.agents import Leader, Follower
from sg_koopman.common.utils import to_numpy, PlotUtilities
from sg_koopman.common.models import FdynBrNet
import sg_koopman.parameters as param
from os import mkdir
from os.path import exists


class FdynTrain:
    def __init__(self) -> None:
        self.dimxl, self.dimul = param.dimxl, param.dimul
        self.dimxf, self.dumuf = param.dimxf, param.dimuf

        # learning parameters
        self.lr = param.nn_fdyn['lr']
        self.mom = param.nn_fdyn['mom']
        self.batch_size = param.nn_fdyn['batch_size']
        self.n_epoch = param.nn_fdyn['n_epoch']
    

    def get_train_data(self, data, N=1000):
        '''
        Prepare interactive trajectory data. data[i, k, :] = [xf_k, uf*_k, xf_kp1, xl_k, ul_k, xl_kp1]
        '''
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
        Generate batch data for training and testing. data[i, k, :] = [xf_k, uf*_k, xf_kp1, xl_k, ul_k, xl_kp1]
        Useful data: [xf, xl, ul, xf_new], only use first horizon, data[:, 0, :]
        '''
        batch_data = []
        n_batch = data.shape[0] // batch_size 
        for i in range(n_batch):
            idx = i * batch_size
            xf = data[idx: idx+batch_size, 0, :self.dimxf]
            xf_new = data[idx: idx+batch_size, 0, self.dimxf+self.dimuf: 2*self.dimxf+self.dimuf]
            xl = data[idx: idx+batch_size, 0, 2*self.dimxf+self.dimuf: 2*self.dimxf+self.dimuf+self.dimxl]
            ul = data[idx: idx+batch_size, 0, 2*self.dimxf+self.dimuf+self.dimxl: 2*self.dimxf+self.dimuf+self.dimxl+self.dimul]
            batch_data.append([xf, xl, ul, xf_new])
        return batch_data
    

    def train_fdynbr(self, Dtrain, Dtest=None, device='cpu'):
        '''
        This function trains a NN for the follower's feedback dybamics: xf_new = f(xf, xl, ul).
        '''
        Dtrain = torch.tensor(Dtrain, device=device)
        Dtest = torch.tensor(Dtest, device=device)
    
        fnet = FdynBrNet()
        fnet.to(device)

        loss_fn = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(fnet.parameters(), lr=self.lr, momentum=self.mom)
        #optimizer = torch.optim.Adam(fnet.parameters(), lr=0.0001)
        train_loss = []

        if Dtest is not None:
            test_loss = np.zeros(self.n_epoch)
            Dtest_batch_data = self.get_batch_data(Dtest, Dtest.shape[0])    # only one-batch test data

        import time
        start_t = time.time()
    
        for t in range(self.n_epoch):
            print("Epoch: {}/{}\n-------------".format(t+1, self.n_epoch))
            Dtrain = Dtrain[torch.randperm(Dtrain.shape[0]), :]     # shuffle training data in each epoch
            batch_data = self.get_batch_data(Dtrain, self.batch_size, device)
            batch_train_loss = np.zeros(len(batch_data))

            for i in range(len(batch_data)):   # iterate over all training data organized by batch
                xf, xl, ul, xf_new = batch_data[i][0], batch_data[i][1], batch_data[i][2], batch_data[i][3]
                loss = loss_fn(fnet(xf, xl, ul), xf_new) / self.batch_size
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_train_loss[i]= loss.item()
            
                if (i+1) % 10 == 0:
                    print('    iter: {}/{}, loss: {:.7f}'.format(i+1, len(batch_data), loss.item()))
            train_loss.append(batch_train_loss)

            # validation/test loss
            if Dtest is not None:
                xf, xl, ul, xf_new = Dtest_batch_data[0][0], Dtest_batch_data[0][1], Dtest_batch_data[0][2], Dtest_batch_data[0][3]
                loss = loss_fn(fnet(xf, xl, ul), xf_new) / Dtest.shape[0]
                test_loss[t] = loss.item()
    
        end_t = time.time()
        print("Elapsed time: {:.3f} min.".format((end_t-start_t)/60))
    
        tl = np.mean(np.vstack(train_loss), axis=1)
    
        # save the follower's trained dynamical model    
        save_flag = True
        if save_flag:
            model_dir_name = 'data/nn_fdynbr/model_N' + str(Dtrain.shape[0]) + '/'
            if not exists(model_dir_name):
                mkdir(model_dir_name)
            torch.save(fnet.state_dict(), model_dir_name+'fdynbr.pt')
            np.save(model_dir_name+'train_loss.npy', tl)
            np.save(model_dir_name+'test_loss.npy', test_loss)

        return fnet.state_dict()


    def predict_error(self, data, fnetdict, N=20, K=5):
        '''
        Compute K-step prediction error (mean and std) using N ground truth data trajectories.
        '''
        data = data[self.rng.choice(data.shape[0], N, replace=False), :]
        data = torch.tensor(data)
        
        fnet = FdynBrNet()
        fnet.load_state_dict(torch.load(fnetdict))
        
        error = torch.zeros((N,K))
        for i in range(N):
            xl_traj = data[i, :, 2*self.dimxf+self.dimuf: 2*self.dimxf+self.dimuf+self.dimxl]  # [xl_0, ..., xl_Km1]
            ul_traj = data[i, :, 2*self.dimxf+self.dimuf+self.dimxl: 2*self.dimxf+self.dimuf+self.dimxl+self.dimul]    # [ul_0, ..., ul_Km1]
            xf_pred = torch.zeros((K, self.dimxf))  # does not contain xf_0
            xf_k = data[i, 0, :self.dimxf]   # xf_0
            for k in range(K):
                xf_pred[k, :] = fnet(xf_k, xl_traj[k], ul_traj[k])  # xf_kp1 = fnet(xf_k, xl_k, ul_k)
                xf_k = xf_pred[k, :]

            xf_traj = data[i, :, self.dimxf+self.dimuf: 2*self.dimxf+self.dimuf]  # [xf_1, xf_2, ..., xf_K]
            error[i, :] = torch.linalg.norm(xf_pred-xf_traj, dim=1)
        err_mean = torch.mean(error, dim=0)
        err_std = torch.std(error, dim=0)
        return err_mean.detach().numpy(), err_std.detach().numpy()


def rh_planning(fmodeldict=None):
    '''
    This function performs receding horizon planning.
    '''
    fnet = FdynBrNet()
    fnet.load_state_dict(fmodeldict)

    leader = Leader()
    follower = Follower()
    util = PlotUtilities()
    xl_traj, ul_traj = [], [] 
    xf_traj, uf_traj = [], []

    xl, xf = leader.xl0, follower.xf0
    xl_traj.append(xl)
    xf_traj.append(xf)

    RC_MAX = 100    # max receding horizon
    X_pre = None
    import time
    for t in range(RC_MAX):
        print(f'-------RC planning iteration: {t+1} -------')
        st = time.time()
        ul, X_pre = leader.ocp_nn(fnet, xl, xf, X_pre)    # or always set X_pre=None and use astar
        et = time.time()
        print(f'  planning time: {(et-st):.4f}s')
        
        xl_new = leader.dyn(xl, ul)
        uf = follower.get_br(xf, xl_new)
        xf_new = follower.dyn(xf, uf)

        ul_traj.append(ul)
        uf_traj.append(uf)
        xl_traj.append(xl)
        xf_traj.append(xf)
        xl = xl_new
        xf = xf_new

        util.plot_traj(np.array(xl_traj), np.array(xf_traj))
        # stop condition: t > RC_MAX or near target
    
    xl_traj, ul_traj = np.array(xl_traj), np.array(ul_traj)
    xf_traj, uf_traj = np.array(xf_traj), np.array(uf_traj)
    
    return xl_traj, ul_traj, xf_traj, uf_traj


class OptimalControl(Leader):
    '''
    Leader's optimal control problem. X = [x_1, x_2, ..., x_T, u_0, ..., u_{T-1}]
    '''
    def __init__(self) -> None:
        super().__init__()


    def ocp(self, fmodel, xl0, xf0, X_pre=None):
        '''
        This function solves the optimal control problem to generate ul.
        '''
        def fdyn(xf, xl, ul):
            xf, xl, ul = torch.tensor(xf), torch.tensor(xl), torch.tensor(ul)
            xf_new = to_numpy( fmodel(xf, xl, ul) )
            return xf_new
        def fdyn_jac(xf, xl, ul):
            xf, xl, ul = torch.tensor(xf), torch.tensor(xl), torch.tensor(ul)
            jac_xf, jac_xl, jac_ul = fmodel.get_input_jac(xf, xl, ul)
            jac_xf = to_numpy( jac_xf )
            jac_xl = to_numpy( jac_xl )
            jac_ul = to_numpy( jac_ul )
            return jac_xf, jac_xl, jac_ul

        constr = []

        # leader's safety constraints: c_i(xl_t) >= 0
        def con_safe(X):
            f = np.zeros(self.dimT*len(self.obs))
            for t in range(self.dimT):
                i_xl = np.arange(t*self.dimx, t*self.dimx+self.dimxl)
                f[t*len(self.obs): (t+1)*len(self.obs)] = self.safety_constr(X[i_xl])
            return f
        def jac_con_safe(X):
            df = np.zeros((self.dimT*len(self.obs), X.shape[0]))
            for t in range(self.dimT):
                i_xl = np.arange(t*self.dimx, t*self.dimx+self.dimxl)
                df[t*len(self.obs): (t+1)*len(self.obs), i_xl] = self.safety_constr_jac(X[i_xl])
            return df
        lb_safe = np.ones(self.dimT*len(self.obs))
        constr.append( NonlinearConstraint(con_safe, 0*lb_safe, np.inf*lb_safe, jac=jac_con_safe) )
    
        # dynamic constraints: x_{t+1} = f(x_t, u_t), f = [fl(xl, ul); ff(xf, xl, ul)], x = [xl; xf]
        def con_dyn(X):
            fl = np.zeros(self.dimT*self.dimxl)
            ff = np.zeros(self.dimT*self.dimxf)
            for t in range(self.dimT):
                # get state and input component
                i_xl = np.arange(t*self.dimx, t*self.dimx+self.dimxl)
                i_xf = np.arange(t*self.dimx+self.dimxl, (t+1)*self.dimx)
                i_ul = np.arange(t*self.dimul, (t+1)*self.dimul) + self.dimT*self.dimx
                xl_tp1, ul_t, xf_tp1 = X[i_xl], X[i_ul], X[i_xf]
                if t == 0:
                    xl_t, xf_t = xl0, xf0
                else:
                    xl_t, xf_t = X[i_xl-self.dimx], X[i_xf-self.dimx]
            
                # compute x_tp1 - f(x, u)
                fl[t*self.dimxl: (t+1)*self.dimxl] = xl_tp1 - self.dyn(xl_t, ul_t)
                ff[t*self.dimxf: (t+1)*self.dimxf] = xf_tp1 - fdyn(xf_t, xl_t, ul_t)
            return np.concatenate( (fl,ff) )
        def jac_con_dyn(X):
            dfl = np.zeros((self.dimT*self.dimxl, X.shape[0]))
            dff = np.zeros((self.dimT*self.dimxf, X.shape[0]))
            for t in range(self.dimT):
                # get state and input component
                i_xl = np.arange(t*self.dimx, t*self.dimx+self.dimxl)
                i_xf = np.arange(t*self.dimx+self.dimxl, (t+1)*self.dimx)
                i_ul = np.arange(t*self.dimul, (t+1)*self.dimul) + self.dimT*self.dimx
                xl_tp1, ul_t, xf_tp1 = X[i_xl], X[i_ul], X[i_xf]
                if t == 0:
                    xl_t, xf_t = xl0, xf0
                else:
                    xl_t, xf_t = X[i_xl-self.dimx], X[i_xf-self.dimx]
            
                # compute jacobian
                dfl_dx, dfl_du = self.dyn_jac(xl_t, ul_t)
                dfl[t*self.dimxl: (t+1)*self.dimxl, i_xl] = np.eye(self.dimxl)
                dfl[t*self.dimxl: (t+1)*self.dimxl, i_ul] = -dfl_du

                dff_dxf, dff_dxl, dff_dul = fdyn_jac(xf_t, xl_t, ul_t)
                dff[t*self.dimxf: (t+1)*self.dimxf, i_xf] = np.eye(self.dimxf)
                dff[t*self.dimxf: (t+1)*self.dimxf, i_ul] = -dff_dul
                if t > 0:
                    dfl[t*self.dimxl: (t+1)*self.dimxl, i_xl-self.dimx] = -dfl_dx

                    dff[t*self.dimxf: (t+1)*self.dimxf, i_xl-self.dimx] = -dff_dxl
                    dff[t*self.dimxf: (t+1)*self.dimxf, i_xf-self.dimx] = -dff_dxf
            return np.vstack( (dfl,dff) )
        lb_dyn = np.zeros(self.dimT*self.dimx)
        constr.append( NonlinearConstraint(con_dyn, lb=lb_dyn, ub=lb_dyn, jac=jac_con_dyn) )
    
        # stay in the square region if necessary
        D = np.block([      # D@x_t = [pl, pf]
            [np.eye(self.dimxl-1), np.zeros((self.dimxl-1, 1+self.dimxf))],
            [np.zeros((self.dimxf-1, self.dimxl)), np.eye(self.dimxf-1), np.zeros((self.dimxf-1, 1))]
        ])  
        DD = np.kron(np.eye(self.dimT), D)
        DD = np.hstack( (DD, np.zeros((self.dimT*(self.dimx-2), self.dimT*self.dimul))) )
        lb_env = np.kron(np.ones(self.dimT), np.ones(self.dimx-2))
        #constr.append(LinearConstraint(DD, 0*lb_env, self.ws_len*lb_env))

        # input constraints
        BB = np.kron(np.eye(self.dimT), np.zeros((self.dimul,self.dimx+self.dimul)))
        BB[:, self.dimT*self.dimx:] = np.kron(np.eye(self.dimT), np.eye(self.dimul))
        lb_a = np.kron(np.ones(self.dimT), np.array([self.vmin, self.wmin]))
        ub_a = np.kron(np.ones(self.dimT), np.array([self.vmax, self.wmax]))
        constr.append(LinearConstraint(BB, lb_a, ub_a))

        # objective function
        def myobj(X):
            J = 0
            for t in range(self.dimT):
                i_xl = np.arange(t*self.dimx, t*self.dimx+self.dimxl)
                i_xf = np.arange(t*self.dimx+self.dimxl, (t+1)*self.dimx)
                i_ul = np.arange(t*self.dimul, (t+1)*self.dimul) + self.dimT*self.dimx
                xl_tp1, ul_t, xf_tp1 = X[i_xl], X[i_ul], X[i_xf]
                J += (xl_tp1 - xf_tp1) @ self.Ql1 @ (xl_tp1 - xf_tp1) + (xl_tp1 - self.xd) @ self.Ql2 @ (xl_tp1 - self.xd) + ul_t @ self.Rl @ ul_t
            return J
        def jac_myobj(X):
            dJ = np.zeros_like(X)
            for t in range(self.dimT):
                i_xl = np.arange(t*self.dimx, t*self.dimx+self.dimxl)
                i_xf = np.arange(t*self.dimx+self.dimxl, (t+1)*self.dimx)
                i_ul = np.arange(t*self.dimul, (t+1)*self.dimul) + self.dimT*self.dimx
                xl_tp1, ul_t, xf_tp1 = X[i_xl], X[i_ul], X[i_xf]
            
                dJ[i_xl] = 2*self.Ql1 @ (xl_tp1 - xf_tp1) + 2*self.Ql2 @ (xl_tp1 - self.xd)
                dJ[i_xf] = -2*self.Ql1 @ (xl_tp1 - xf_tp1)
                dJ[i_ul] = 2*self.Rl @ ul_t
            return dJ
        def myobj_s2(X):
            J = 0
            for t in range(self.dimT):
                i_xl = np.arange(t*self.dimx, t*self.dimx+self.dimxl)
                i_xf = np.arange(t*self.dimx+self.dimxl, (t+1)*self.dimx)
                i_ul = np.arange(t*self.dimul, (t+1)*self.dimul) + self.dimT*self.dimx
                xl_tp1, ul_t, xf_tp1 = X[i_xl], X[i_ul], X[i_xf]
                J += (xl_tp1 - xf_tp1) @ self.Ql1_s2 @ (xl_tp1 - xf_tp1) + (xl_tp1 - self.xd) @ self.Ql2_s2 @ (xl_tp1 - self.xd) + ul_t @ self.Rl_s2 @ ul_t
            return J
        def jac_myobj_s2(X):
            dJ = np.zeros_like(X)
            for t in range(self.dimT):
                i_xl = np.arange(t*self.dimx, t*self.dimx+self.dimxl)
                i_xf = np.arange(t*self.dimx+self.dimxl, (t+1)*self.dimx)
                i_ul = np.arange(t*self.dimul, (t+1)*self.dimul) + self.dimT*self.dimx
                xl_tp1, ul_t, xf_tp1 = X[i_xl], X[i_ul], X[i_xf]
            
                dJ[i_xl] = 2*self.Ql1_s2 @ (xl_tp1 - xf_tp1) + 2*self.Ql2_s2 @ (xl_tp1 - self.xd)
                dJ[i_xf] = -2*self.Ql1_s2 @ (xl_tp1 - xf_tp1)
                dJ[i_ul] = 2*self.Rl_s2 @ ul_t
            return dJ
    
        '''
        #---------- test jac -----------#
        from scipy.optimize import approx_fprime, check_grad
        tmp = np.zeros(10)
        for ii in range(tmp.shape[0]):
            X00 = np.random.rand(X0.shape[0]) * 5 - 1
            #tmp[ii] = check_grad(myobj, jac_myobj, X00)
            tmp[ii] = check_grad(con_dyn, jac_con_dyn, X00)
            #tmp[ii] = check_grad(con_safe, jac_con_safe, X00)
            #tmp[ii] = check_grad(con_a, jac_con_a, X00)
        #print(tmp)
        #a1 = approx_fprime(X0, test_Jf)
        #a1 = a1[self.dimT*(self.dimxl+self.dimul+self.dimxf): ]
        #a2 = np.abs(a1 - con_dJfduf(X0))
        #print(np.linalg.norm(a2), a2)
        #-------------------------------#
        '''

        sce_th = 1
        if np.linalg.norm(xl0[:2]-xf0[:2]) <= sce_th or X_pre is None:   # scenario 1, near the follower, guide to target
            X0 = self.init_1(xl0, xf0)
            res = minimize(myobj, X0, jac=jac_myobj, constraints=constr, options={'maxiter': 200, 'disp': False})
            print('  scenario 1: status {}: {}'.format(res.status, res.message))
        else:   # scenario 2, far from the follower, go to the follower
            X0 = self.init_2(xl0, xf0)
            #X0 = self.init_shiftX(X_pre)
            res = minimize(myobj_s2, X0, jac=jac_myobj_s2, constraints=constr, options={'maxiter': 200, 'disp': False})
            print('  scenario 2: status {}: {}'.format(res.status, res.message))
        
        # convert results to trajectory
        ul_opt = res.x[self.dimT*self.dimx: ]
        ul_opt = ul_opt.reshape((self.dimT, self.dimul))
        print(f'  ulopt: {ul_opt[0, :]}')
        
        return ul_opt[0, :], res.x


    def init_1(self, xl0, xf0):
        '''
        Initialize trajectory for leader to guide follower to target. For scenario 1.
        '''
        dxdy = param.dxdy
        grid_map = param.grid_map

        # planning trajectory with Astar
        xl_traj = self.astar_x_traj(xl0, self.xd, grid_map, dxdy, horizon=self.dimT+1)    # xl_traj[0,:] = xl0
        xf_traj = self.astar_x_traj(xf0, xl0, grid_map, dxdy, horizon=self.dimT+1)
        ul_traj = np.zeros((self.dimT, self.dimul)) + 0.1*self.rng.random((self.dimT, self.dimul))
        x_traj = np.hstack( (xl_traj, xf_traj) )
        X0 = np.concatenate( (x_traj[1:,:].flatten(), ul_traj.flatten()) )
        return X0
    

    def init_2(self, xl0, xf0):
        '''
        Initialize trajectory for leader to go to follower. For scenario 2.
        '''
        dxdy = param.dxdy
        grid_map = param.grid_map

        # planning trajectory with Astar
        xl_traj = self.astar_x_traj(xl0, xf0, grid_map, dxdy, horizon=self.dimT+1)    # xl_traj[0,:] = xl0
        xf_traj = self.astar_x_traj(xf0, xl0, grid_map, dxdy, horizon=self.dimT+1)
        ul_traj = np.zeros((self.dimT, self.dimul)) + 0.1*self.rng.random((self.dimT, self.dimul))
        x_traj = np.hstack( (xl_traj, xf_traj) )
        X0 = np.concatenate( (x_traj[1:,:].flatten(), ul_traj.flatten()) )
        return X0


    def init_shiftX(self, X_pre):
        '''
        This function initializes X given the previous solution of OCP.
        Set X[-1] = xf and U[-1] = 0. The rest shift left.
        '''
        X0 = np.zeros_like(X_pre)
        xf = np.concatenate( (self.xd, self.xd) )
        
        i_x = np.arange((self.dimT-1)*self.dimx)
        X0[i_x] = X_pre[i_x+self.dimx]
        X0[(self.dimT-1)*self.dimx: self.dimT*self.dimx] = xf
        
        i_ul = np.arange((self.dimT-1)*self.dimul) + self.dimT*self.dimx
        X0[i_ul] = X_pre[i_ul+self.dimul]
        
        return X0
