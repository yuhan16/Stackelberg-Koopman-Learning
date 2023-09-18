"""
This module implements DMD to estimate feedback dynamics and perform receding horizon planning.
"""
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint
from sg_koopman.common.agents import Leader, Follower
from sg_koopman.common.utils import PlotUtilities
import sg_koopman.parameters as param
from os import mkdir
from os.path import exists


class DMDTrain:
    '''
    Use DMD to find the follower's feedback dynamics xf_new = A @ xf + B1 @ xl + B2 @ ul
    '''
    def __init__(self) -> None:
        self.dimxl, self.dimul = param.dimxl, param.dimul
        self.dimxf, self.dumuf = param.dimxf, param.dimuf
        self.gam = param.dmd['gam']
        self.pred_horizon = param.dmd['pred_horizon']
    
    
    def find_dmd_kp(self, data, N=1000):    # linear approximation of feedback dynamics
        if data is None or len(data=0):
            if exists('data/follower_dynbr_lmix.npy'):
                raise Exception('No data provided or found.')
            else:
                data = np.load('data/follower_dynbr_lmix.npy')
    
        if N <= data.shape[0]:
            data = data[np.random.choice(data.shape[0], N, replace=False), :]
        else:
            raise Exception('N exceeds totoal number of data samples.')
    
        K = data.shape[1]
        G, P = 0, 0
        for k in range(self.pred_horizon):
            xf_kp1 = data[:,k, self.dimxf+self.dimuf: 2*self.dimxf+self.dimuf]
            xf_k = data[:, k, 0: self.dimxf]
            xl_k = data[:, k, 2*self.dimxf+self.dimuf: 2*self.dimxf+self.dimuf+self.dimxl]
            ul_k = data[:, k, 2*self.dimxf+self.dimuf+self.dimxl: 2*self.dimxf+self.dimuf+self.dimxl+self.dimul]

            #y_k = np.concatenate( (xf_k, xl_k, ul_k, np.ones((N,1))), axis=1 )     # consider bias term
            y_k = np.concatenate( (xf_k, xl_k, ul_k), axis=1 )
            G += y_k.T @ y_k / N * (self.gam**k)
            P += xf_kp1.T @ y_k / N * (self.gam**k)

        Kp = P @ np.linalg.pinv(G)
    
        save_flag = True
        if save_flag:
            model_dir_name = 'data/dmd/model_N' + str(data.shape[0]) + '/'
            if not exists(model_dir_name):
                mkdir(model_dir_name)
            np.save(model_dir_name+'Kp.npy', Kp)
        
        return Kp


    def predict_error(self, data, Kp, N=20, K=10):
        data = data[self.rng.choice(data.shape[0], N, replace=False), :]
        error = np.zeros((N,K))
    
        for i in range(N):
            xl_traj = data[i, :, 2*self.dimxf+self.dimuf: 2*self.dimxf+self.dimuf+self.dimxl]  # [xl_0, ..., xl_Km1]
            ul_traj = data[i, :, 2*self.dimxf+self.dimuf+self.dimxl: 2*self.dimxf+self.dimuf+self.dimxl+self.dimul]    # [ul_0, ..., ul_Km1]
            xf_pred = np.zeros((K, self.dimxf))     # does not contain xf_0
            xf_k = data[i, 0, :self.dimxf]   # xf_0
            for k in range(K):
                #xf_pred[k] = Kp @ np.concatenate( (xf_k, xl_k, ul_k, np.ones(1)) )
                xf_pred[k] = Kp @ np.concatenate( (xf_k, xl_traj[k], ul_traj[k]) )
                xf_k = xf_pred[k]
            
            xf_traj = data[i, :, self.dimxf+self.dimuf: 2*self.dimxf+self.dimuf]   # [xf_1, xf_2, ..., xf_K]
            error[i, :] = np.linalg.norm(xf_pred - xf_traj, axis=1)
            
        err_mean = np.mean(error, axis=0)
        err_std = np.std(error, axis=0)
        return err_mean, err_std
    

def rh_planning(Kp):
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
        print(f'-------RH planning iteration: {t+1} -------')
        st = time.time()
        ul, X_pre = ocp_solver.ocp(Kp, xl, xf, X_pre)
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

        pltutil.plot_traj(np.array(xl_traj), np.array(xf_traj))
        # stop condition: t > RC_MAX or near target
    
    xl_traj, ul_traj = np.array(xl_traj), np.array(ul_traj)
    xf_traj, uf_traj = np.array(xf_traj), np.array(uf_traj)
    
    return xl_traj, ul_traj, xf_traj, uf_traj


class OptimalControl(Leader):
    '''
    Leader's optimal control problem. X = [xl_1,xl_2,..., xf_1,xf_2,..., ul_1,ul_2...]
    '''
    def __init__(self) -> None:
        super().__init__()


    def ocp(self, Kp, xl0, xf0, X_pre=None):
        constr = []

        A, B1, B2 = Kp[:, : self.dimxf], Kp[:, self.dimxf: self.dimxf+self.dimxl], Kp[:, self.dimxf+self.dimxl: -1]
        # follower's linear dynamics: xf_kp1 = A @ xf_k + B1 @ xl_k + B2 @ ul_k
        tmp = np.diag(np.ones(self.dimT-1), k=-1)
        aa1 = np.kron(np.eye(self.dimT), -np.eye(self.dimxf))
        aa2 = np.kron(tmp, A)
        AA = aa1 + aa2
        BB1 = np.kron(tmp, B1)
        BB2 = np.kron(np.eye(self.dimT), B2)
        bigA = np.hstack( (BB1, AA, BB2) )
        b = np.zeros(self.dimT*self.dimxf)
        b[: self.dimxf] = - A @ xf0 - B1 @ xl0
        constr.append(LinearConstraint(bigA, lb=b, ub=b))

        def myobj(X):
            J = 0
            for t in range(self.dimT):
                i_xl = np.arange(t*self.dimxl, (t+1)*self.dimxl)
                i_y = np.arange(t*self.dimxf, (t+1)*self.dimxf) + self.dimT*self.dimxl
                i_ul = np.arange(t*self.dimul, (t+1)*self.dimul) + self.dimT*(self.dimxl+self.dimxf)
                xl_tp1, y_tp1, ul_t = X[i_xl], X[i_y], X[i_ul]
                J += (xl_tp1 - y_tp1) @ self.Ql1 @ (xl_tp1 - y_tp1) + (xl_tp1 - self.xd) @ self.Ql2 @ (xl_tp1 - self.xd) + ul_t @ self.Rl @ ul_t
            return J
        def jac_myobj(X):
            dJ = np.zeros_like(X)
            for t in range(self.dimT):
                i_xl = np.arange(t*self.dimxl, (t+1)*self.dimxl)
                i_y = np.arange(t*self.dimxf, (t+1)*self.dimxf) + self.dimT*self.dimxl
                i_ul = np.arange(t*self.dimul, (t+1)*self.dimul) + self.dimT*(self.dimxl+self.dimxf)
                xl_tp1, y_tp1, ul_t = X[i_xl], X[i_y], X[i_ul]
                
                dJ[i_xl] = 2*self.Ql1 @ (xl_tp1 - y_tp1) + 2*self.Ql2 @ (xl_tp1 - self.xd)
                dJ[i_y] = -2*self.Ql1 @ (xl_tp1 - y_tp1)
                dJ[i_ul] = 2*self.Rl @ ul_t
            return dJ
        def myobj_s2(X):
            J = 0
            for t in range(self.dimT):
                i_xl = np.arange(t*self.dimxl, (t+1)*self.dimxl)
                i_y = np.arange(t*self.dimxf, (t+1)*self.dimxf) + self.dimT*self.dimxl
                i_ul = np.arange(t*self.dimul, (t+1)*self.dimul) + self.dimT*(self.dimxl+self.dimxf)
                xl_tp1, y_tp1, ul_t = X[i_xl], X[i_y], X[i_ul]
                J += (xl_tp1 - y_tp1) @ self.Ql1_s2 @ (xl_tp1 - y_tp1) + (xl_tp1 - self.xd) @ self.Ql2_s2 @ (xl_tp1 - self.xd) + ul_t @ self.Rl_s2 @ ul_t
            return J
        def jac_myobj_s2(X):
            dJ = np.zeros_like(X)
            for t in range(self.dimT):
                i_xl = np.arange(t*self.dimxl, (t+1)*self.dimxl)
                i_y = np.arange(t*self.dimxf, (t+1)*self.dimxf) + self.dimT*self.dimxl
                i_ul = np.arange(t*self.dimul, (t+1)*self.dimul) + self.dimT*(self.dimxl+self.dimxf)
                xl_tp1, y_tp1, ul_t = X[i_xl], X[i_y], X[i_ul]
                
                dJ[i_xl] = 2*self.Ql1_s2 @ (xl_tp1 - y_tp1) + 2*self.Ql2_s2 @ (xl_tp1 - self.xd)
                dJ[i_y] = -2*self.Ql1_s2 @ (xl_tp1 - y_tp1)
                dJ[i_ul] = 2*self.Rl_s2 @ ul_t
            return dJ

        def con_dyn(X):     # leader's dynamics only xl_tp1 = fL(xl_t, ul_t)
            f = np.zeros(self.dimT*self.dimxl)
            for t in range(self.dimT):
                i_xl = np.arange(t*self.dimxl, (t+1)*self.dimxl)
                i_ul = np.arange(t*self.dimul, (t+1)*self.dimul) + self.dimT*(self.dimxl+self.dimxf)
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
                i_ul = np.arange(t*self.dimul, (t+1)*self.dimul) + self.dimT*(self.dimxl+self.dimxf)
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
        constr.append( NonlinearConstraint(con_safe, 0*lb_safe, np.inf*lb_safe, jac=jac_con_safe) )

        # leader's input constraints
        BB = np.kron(np.eye(self.dimT), np.zeros((self.dimul,self.dimxl+self.dimul+self.dimxf)))
        BB[:, self.dimT*(self.dimxl+self.dimxf): ] = np.kron(np.eye(self.dimT), np.eye(self.dimul))
        lb_a = np.kron(np.ones(self.dimT), np.array([self.vmin, self.wmin]))
        ub_a = np.kron(np.ones(self.dimT), np.array([self.vmax, self.wmax]))
        constr.append(LinearConstraint(BB, lb_a, ub_a))

        '''
        #---------- test jac -----------#
        from scipy.optimize import approx_fprime, check_grad
        tmp = np.zeros(10)
        for ii in range(tmp.shape[0]):
            X00 = np.random.rand(self.dimT*(self.dimxl+self.dimxf+self.dimul)) * 5 - 1
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
            X0 = self.init_1(xl0, xf0)
            res = minimize(myobj, X0, jac=jac_myobj, constraints=constr, options={'maxiter': 200, 'disp': False})
            print('  scenario 1: status {}: {}'.format(res.status, res.message))
        else:   # scenario 2, far from the follower, go to the follower
            X0 = self.init_2(xl0, xf0)
            #X0 = self.init_shiftX(X_pre)
            res = minimize(myobj_s2, X0, jac=jac_myobj_s2, constraints=constr, options={'maxiter': 200, 'disp': False})
            print('  scenario 2: status {}: {}'.format(res.status, res.message))

        # convert results to trajectory
        ul_opt = res.x[self.dimT*(self.dimxl+self.dimxf): ]
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

        # construct X0
        X0 = np.zeros(self.dimT*(self.dimxl+self.dimul+self.dimxf))     # X0 = [xl, xf, ul]
        tmp = np.concatenate( (xl_traj[1:,:].flatten(), xf_traj[1:,:].flatten()) )
        X0[: tmp.size] = tmp
        X0[tmp.size:] = 0*np.random.rand(self.dimul*self.dimT)

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

        # construct X0
        X0 = np.zeros(self.dimT*(self.dimxl+self.dimul+self.dimxf))     # X0 = [xl, xf, ul]
        tmp = np.concatenate( (xl_traj[1:,:].flatten(), xf_traj[1:,:].flatten()) )
        X0[: tmp.size] = tmp
        X0[tmp.size:] = 0*np.random.rand(self.dimul*self.dimT)

        return X0


    def init_shiftX(self, X_pre):
        '''
        This function initializes X given the previous solution of OCP.
        Set X[-1] = xd and U[-1] = 0. The rest shift left.
        '''
        X0 = np.zeros_like(X_pre)
    
        i_xl = np.arange((self.dimT-1)*self.dimxl)
        X0[i_xl] = X_pre[i_xl+self.dimxl]
        X0[i_xl[-1]+1: i_xl[-1]+1+self.dimxl] = self.xd

        i_xf = np.arange((self.dimT-1)*self.dimxf) + self.dimT*self.dimxl
        X0[i_xf] = X_pre[i_xf+self.dimxf]
        X0[i_xf[-1]+1: i_xf[-1]+1+self.dimxf] = self.xd
    
        i_ul = np.arange((self.dimT-1)*self.dimul) + self.dimT*(self.dimxl+self.dimxf)
        X0[i_ul] = X_pre[i_ul+self.dimul]
    
        return X0
