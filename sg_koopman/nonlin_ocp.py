"""
This script implements nonlinear optimization to perform receding horizon planning.
"""
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint
from common.agents import Leader, Follower
from common.utils import PlotUtilities
import sg_koopman.parameters as param
import time


def rh_planning():
    """
    This function performs receding horizon planning.
    """
    leader = Leader()
    follower = Follower()
    ocp_solver = OptimalControl()
    util = PlotUtilities()
    xl_traj, ul_traj = [], [] 
    xf_traj, uf_traj = [], []

    xl, xf = leader.xl0, follower.xf0
    xl_traj.append(xl)
    xf_traj.append(xf)

    RC_MAX = 200    # max receding horizon
    X_pre = None
    for t in range(RC_MAX):
        print(f'------- RH planning iteration: {t+1} -------')
        st = time.time()
        ul, X_pre = ocp_solver.ocp_nonlin(xl, xf, X_pre)
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
    Leader's optimal control problem. X = [xl_1, ..., xl_T, ul_0, ..., ul_{T-1}, xf_1, ..., xf_T, uf_0, ..., uf_{T-1}]
    '''
    def __init__(self) -> None:
        super().__init__()
    

    def ocp_nonlin(self, xl0, xf0, X_pre=None):
        '''
        This function solves a nonlinear optimization problem to generate ul and uf.
        '''
        follower = Follower()
        constr = []

        def con_dyn(X):  # leader and follower's dynamics
            fl = np.zeros(self.dimT*self.dimxl)
            ff = np.zeros(self.dimT*self.dimxf)
            for t in range(self.dimT):
                # get state and input component
                i_xl = np.arange(t*self.dimxl, (t+1)*self.dimxl)
                i_ul = np.arange(t*self.dimul, (t+1)*self.dimul) + self.dimT*self.dimxl
                i_xf = np.arange(t*self.dimxf, (t+1)*self.dimxf) + self.dimT*(self.dimxl+self.dimul)
                i_uf = np.arange(t*self.dimuf, (t+1)*self.dimuf) + self.dimT*(self.dimxl+self.dimul+self.dimxf)
                xl_tp1, ul_t, xf_tp1, uf_t = X[i_xl], X[i_ul], X[i_xf], X[i_uf]
                if t == 0:
                    xl_t, xf_t = xl0, xf0
                else:
                    xl_t, xf_t = X[i_xl-self.dimxl], X[i_xf-self.dimxf]
            
                # compute x_tp1 - f(x, u)
                fl[t*self.dimxl: (t+1)*self.dimxl] = xl_tp1 - self.dyn(xl_t, ul_t)
                ff[t*self.dimxf: (t+1)*self.dimxf] = xf_tp1 - follower.dyn(xf_t, uf_t)
            return np.concatenate( (fl,ff) )
        def jac_con_dyn(X):
            dfl = np.zeros((self.dimT*self.dimxl, X.shape[0]))
            dff = np.zeros((self.dimT*self.dimxf, X.shape[0]))
            for t in range(self.dimT):
                # get state and input component
                i_xl = np.arange(t*self.dimxl, (t+1)*self.dimxl)
                i_ul = np.arange(t*self.dimul, (t+1)*self.dimul) + self.dimT*self.dimxl
                i_xf = np.arange(t*self.dimxf, (t+1)*self.dimxf) + self.dimT*(self.dimxl+self.dimul)
                i_uf = np.arange(t*self.dimuf, (t+1)*self.dimuf) + self.dimT*(self.dimxl+self.dimul+self.dimxf)
                xl_tp1, ul_t, xf_tp1, uf_t = X[i_xl], X[i_ul], X[i_xf], X[i_uf]
                if t == 0:
                    xl_t, xf_t = xl0, xf0
                else:
                    xl_t, xf_t = X[i_xl-self.dimxl], X[i_xf-self.dimxf]
            
                # compute jacobian
                dfl_dx, dfl_du = self.dyn_jac(xl_t, ul_t)
                dff_dx, dff_du = follower.dyn_jac(xf_t, uf_t)
                dfl[t*self.dimxl: (t+1)*self.dimxl, i_xl] = np.eye(self.dimxl)
                dfl[t*self.dimxl: (t+1)*self.dimxl, i_ul] = -dfl_du
                dff[t*self.dimxf: (t+1)*self.dimxf, i_xf] = np.eye(self.dimxf)
                dff[t*self.dimxf: (t+1)*self.dimxf, i_uf] = -dff_du
                if t > 0:
                    dfl[t*self.dimxl: (t+1)*self.dimxl, i_xl-self.dimxl] = -dfl_dx
                    dff[t*self.dimxf: (t+1)*self.dimxf, i_xf-self.dimxf] = -dff_dx
            return np.vstack( (dfl,dff) )
        lb_dyn = np.zeros(self.dimT*self.dimx)
        constr.append( NonlinearConstraint(con_dyn, lb=lb_dyn, ub=lb_dyn, jac=jac_con_dyn) )
    
        def con_safe(X):    # leader's safety constraint c_i(xl) >= 0
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
    
        def test_Jf(X):     # used to test con_dJfduf
            J = 0
            mu = 1
            for t in range(self.dimT):
                i_xl = np.arange(t*self.dimxl, (t+1)*self.dimxl)
                i_xf = np.arange(t*self.dimxf, (t+1)*self.dimxf) + self.dimT*(self.dimxl+self.dimul)
                i_uf = np.arange(t*self.dimuf, (t+1)*self.dimuf) + self.dimT*(self.dimxl+self.dimul+self.dimxf)
                xl_tp1, xf_tp1, uf_t = X[i_xl], X[i_xf], X[i_uf]
                if t == 0:
                    xf_t = xf0
                else:
                    xf_t = X[i_xf-self.dimxf]
            
                J += (xl_tp1 - follower.dyn(xf_t, uf_t)) @ follower.Qf1 @ (xl_tp1 - follower.dyn(xf_t, uf_t)) \
                        + (follower.dyn(xf_t, uf_t) - follower.xd) @ follower.Qf2 @ (follower.dyn(xf_t, uf_t) - follower.xd) \
                            + uf_t @ follower.Rf @ uf_t
                J+= (-mu) * np.log(follower.safety_constr( follower.dyn(xf_t, uf_t) )).sum()
            return J
        def con_dJfduf(X):
            '''
            Jf_t = |xl_tp1 - f(xf_t, uf_t)| + |f(xf_t, uf_t) - xd| + |uf_t| - mu * \sum_i log[c_i(f(xf_t, uf_t))]
            only take derivative of uf_t
            '''
            f = np.zeros(self.dimT*self.dimuf)
            mu = 1
            for t in range(self.dimT):
                i_xl = np.arange(t*self.dimxl, (t+1)*self.dimxl)
                i_xf = np.arange(t*self.dimxf, (t+1)*self.dimxf) + self.dimT*(self.dimxl+self.dimul)
                i_uf = np.arange(t*self.dimuf, (t+1)*self.dimuf) + self.dimT*(self.dimxl+self.dimul+self.dimxf)
                xl_tp1, xf_tp1, uf_t = X[i_xl], X[i_xf], X[i_uf]
                if t == 0:
                    xf_t = xf0
                else:
                    xf_t = X[i_xf-self.dimxf]

                dx, du = follower.dyn_jac(xf_t, uf_t)
                c = follower.safety_constr(follower.dyn(xf_t, uf_t))
                dc = follower.safety_constr_jac(follower.dyn(xf_t, uf_t))

                #f[i_xl] = 2*(xl_tp1 - follower.dyn(xf_t, uf_t)) @ follower.Qf1
                f[t*self.dimuf: (t+1)*self.dimuf] = 2*(xl_tp1 - follower.dyn(xf_t, uf_t)) @ follower.Qf1 @ (-du) \
                    + 2*(follower.dyn(xf_t, uf_t) - follower.xd) @ follower.Qf2 @ du + 2*follower.Rf @ uf_t
                for i in range(len(self.obs)):
                    f[t*self.dimuf: (t+1)*self.dimuf] -= mu / c[i] * dc[i] @ du
            return f
        lb_dJf = np.zeros(self.dimT*self.dimuf)
        constr.append( NonlinearConstraint(con_dJfduf, lb=lb_dJf, ub=lb_dJf) )
    
        def myobj(X):
            J = 0
            for t in range(self.dimT):
                i_xl = np.arange(t*self.dimxl, (t+1)*self.dimxl)
                i_xf = np.arange(t*self.dimxf, (t+1)*self.dimxf) + self.dimT*(self.dimxl+self.dimul)
                i_ul = np.arange(t*self.dimul, (t+1)*self.dimul) + self.dimT*self.dimxl
                xl_tp1, xf_tp1, ul_t = X[i_xl], X[i_xf], X[i_ul]
                J += (xl_tp1 - xf_tp1) @ self.Ql1 @ (xl_tp1 - xf_tp1) + (xl_tp1 - self.xd) @ self.Ql2 @ (xl_tp1 - self.xd) + ul_t @ self.Rl @ ul_t
            return J
        def jac_myobj(X):
            dJ = np.zeros_like(X)
            for t in range(self.dimT):
                i_xl = np.arange(t*self.dimxl, (t+1)*self.dimxl)
                i_xf = np.arange(t*self.dimxf, (t+1)*self.dimxf) + self.dimT*(self.dimxl+self.dimul)
                i_ul = np.arange(t*self.dimul, (t+1)*self.dimul) + self.dimT*self.dimxl
                xl_tp1, xf_tp1, ul_t = X[i_xl], X[i_xf], X[i_ul]
            
                dJ[i_xl] = 2*self.Ql1 @ (xl_tp1 - xf_tp1) + 2*self.Ql2 @ (xl_tp1 - self.xd)
                dJ[i_xf] = -2*self.Ql1 @ (xl_tp1 - xf_tp1)
                dJ[i_ul] = 2*self.Rl @ ul_t
            return dJ
        def myobj_s2(X):
            J = 0
            for t in range(self.dimT):
                i_xl = np.arange(t*self.dimxl, (t+1)*self.dimxl)
                i_xf = np.arange(t*self.dimxf, (t+1)*self.dimxf) + self.dimT*(self.dimxl+self.dimul)
                i_ul = np.arange(t*self.dimul, (t+1)*self.dimul) + self.dimT*self.dimxl
                xl_tp1, xf_tp1, ul_t = X[i_xl], X[i_xf], X[i_ul]
                J += (xl_tp1 - xf_tp1) @ self.Ql1_s2 @ (xl_tp1 - xf_tp1) + (xl_tp1 - self.xd) @ self.Ql2_s2 @ (xl_tp1 - self.xd) + ul_t @ self.Rl @ ul_t
            return J
        def jac_myobj_s2(X):
            dJ = np.zeros_like(X)
            for t in range(self.dimT):
                i_xl = np.arange(t*self.dimxl, (t+1)*self.dimxl)
                i_xf = np.arange(t*self.dimxf, (t+1)*self.dimxf) + self.dimT*(self.dimxl+self.dimul)
                i_ul = np.arange(t*self.dimul, (t+1)*self.dimul) + self.dimT*self.dimxl
                xl_tp1, xf_tp1, ul_t = X[i_xl], X[i_xf], X[i_ul]
            
                dJ[i_xl] = 2*self.Ql1_s2 @ (xl_tp1 - xf_tp1) + 2*self.Ql2_s2 @ (xl_tp1 - self.xd)
                dJ[i_xf] = -2*self.Ql1_s2 @ (xl_tp1 - xf_tp1)
                dJ[i_ul] = 2*self.Rl @ ul_t
            return dJ
    
        # input constraints
        BB = np.kron(np.eye(self.dimT), np.zeros((self.dimul,self.dimxl+self.dimul+self.dimxf+self.dimuf)))
        BB[:, self.dimT*self.dimxl: self.dimT*(self.dimxl+self.dimul)] = np.kron(np.eye(self.dimT), np.eye(self.dimul))
        lb_a = np.kron(np.ones(self.dimT), np.array([self.vmin, self.wmin]))
        ub_a = np.kron(np.ones(self.dimT), np.array([self.vmax, self.wmax]))
        constr.append(LinearConstraint(BB, lb_a, ub_a))

        '''
        #---------- test jac -----------#
        from scipy.optimize import approx_fprime, check_grad
        tmp = np.zeros(10)
        for ii in range(tmp.shape[0]):
            X00 = np.random.rand(self.dimT*(self.dimx+self.dimul+self.dimuf)) * 5 - 1
            #tmp[ii] = check_grad(myobj, jac_myobj, X00)
            tmp[ii] = check_grad(myobj_s2, jac_myobj_s2, X00)
            #tmp[ii] = check_grad(con_dyn, jac_con_dyn, X00)
            #tmp[ii] = check_grad(con_safe, jac_con_safe, X00)
        print(tmp)
        #a1 = approx_fprime(X0, test_Jf)
        #a1 = a1[self.dimT*(self.dimxl+self.dimul+self.dimxf): ]
        #a2 = np.abs(a1 - con_dJfduf(X0))
        #print(np.linalg.norm(a2), a2)
        #-------------------------------#
        '''
    
        # select scenario
        sce_th = 1
        if np.linalg.norm(xl0[:2]-xf0[:2]) <= sce_th:   # scenario 1, near the follower, guide to target
            X0 = self.init_1(xl0, xf0)
            res = minimize(myobj, X0, jac=jac_myobj, constraints=constr, options={'maxiter': 200, 'disp': False})
            print('  scenario 1: status {}: {}'.format(res.status, res.message))
        else:       # scenario 2, far from the follower, go to the follower
            X0 = self.init_2(xl0, xf0)
            res = minimize(myobj_s2, X0, jac=jac_myobj_s2, constraints=constr, options={'maxiter': 200, 'disp': False})
            print('  scenario 2: status {}: {}'.format(res.status, res.message))
    
        # convert results to trajectory
        ul_opt = res.x[self.dimT*self.dimxl: self.dimT*(self.dimxl+self.dimul)]
        ul_opt = ul_opt.reshape((self.dimT, self.dimul))
        print(f'  ulopt: {ul_opt[0, :]}')
    
        return ul_opt[0, :], res.x#, sce_id


    def init_1(self, xl0, xf0):
        '''
        Initialization for scenario 1, leader to guide follower to target.
        '''
        dxdy = param.dxdy
        grid_map = param.grid_map

        # planning trajectory with Astar
        xl_traj = self.astar_x_traj(xl0, self.xd, grid_map, dxdy, horizon=self.dimT+1)    # xl_traj[0,:] = xl0
        xf_traj = self.astar_x_traj(xf0, xl0, grid_map, dxdy, horizon=self.dimT+1)

        ul_traj = np.zeros((self.dimT, self.dimul)) + 0.1*self.rng.random((self.dimT, self.dimul))
        uf_traj = np.zeros((self.dimT, self.dimuf)) + 0.1*self.rng.random((self.dimT, self.dimuf))

        X0 = np.concatenate( (xl_traj[1:,:].flatten(), ul_traj.flatten(), xf_traj[1:,:].flatten(), uf_traj.flatten()) )
        return X0


    def init_2(self, xl0, xf0):
        '''
        Initialization for scenario 2. leader to meet follower.
        '''
        dxdy = param.dxdy
        grid_map = param.grid_map
        # planning trajectory with Astar

        xl_traj = self.astar_x_traj(xl0, xf0, grid_map, dxdy, horizon=self.dimT+1)
        xf_traj = self.astar_x_traj(xf0, xl0, grid_map, dxdy, horizon=self.dimT+1)

        ul_traj = np.zeros((self.dimT, self.dimul)) + 0.1*self.rng.random((self.dimT, self.dimul))
        uf_traj = np.zeros((self.dimT, self.dimuf)) + 0.1*self.rng.random((self.dimT, self.dimuf))

        X0 = np.concatenate( (xl_traj[1:,:].flatten(), ul_traj.flatten(), xf_traj[1:,:].flatten(), uf_traj.flatten()) )
        return X0


    def init_shiftX(self, X_pre):
        '''
        This function initializes X given the previous solution of OCP.
        Set X[-1] = xf and U[-1] = 0. The rest shift left.
        '''
        X0 = np.zeros_like(X_pre)
    
        i_xl = np.arange((param.dimT-1)*param.dimxl)
        X0[i_xl] = X_pre[i_xl+param.dimxl]
        X0[i_xl[-1]+1: i_xl[-1]+1+param.dimxl] = param.xd

        i_ul = np.arange((param.dimT-1)*param.dimul) + param.dimT*param.dimxl
        X0[i_ul] = X_pre[i_ul+param.dimul]

        i_xf = np.arange((param.dimT-1)*param.dimxl) + param.dimT*(param.dimxl+param.dimul)
        X0[i_xf] = X_pre[i_xf+param.dimxf]
        X0[i_xf[-1]+1: i_xf[-1]+1+param.dimxf] = param.xd
    
        i_uf = np.arange((param.dimT-1)*param.dimuf) + param.dimT*(param.dimxl+param.dimul+param.dimxf)
        X0[i_uf] = X_pre[i_uf+param.dimuf]
    
        return X0
