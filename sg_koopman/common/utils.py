"""
This module implements sampling and plot utilities.
"""
import numpy as np
import torch
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import sg_koopman.parameters as param


def to_torch(x):
    return torch.from_numpy(x).double()


def to_numpy(x):
    return x.detach().cpu().numpy().astype(float)



class SamplingUtilities():
    '''
    Utilities for sampling the trajectory data
    '''
    def __init__(self) -> None:
        self.obs = param.obs_spec
        self.ws_len = param.ws_len

    
    def sample_leader_dyn_random(self, ll, N, K):
        '''
        Randomly sample leader's dynamical trajectory. 
        data[i, k, :] = [xl_k, ul_k, xl_kp1].
        '''
        data = np.zeros( (N,K, 2*ll.dimxl+ll.dimul) )
        for i in range(N):
            print(f'Sample data point {i+1}/{N}...')
            x = ll.feas_x(xc=None, r=None)
            for k in range(K):
                u = ll.feas_u(x)
                x_new = ll.dyn(x, u)
                data[i, k, :] = np.concatenate( (x,u,x_new) )
                x = x_new
        return data


    def sample_leader_dyn_astar(self, ll, N, K):
        '''
        Generate dynamic trajectory using astar path as initial condition. obj only u and xd.
        data[i, k, :] = [xl_k, ul_k, xl_kp1].
        '''
        def myobj(X):   # X = [xl1, ..., xl_K, ul_0, ..., ul_Km1]
            J = 0
            for t in range(K):
                i_x = np.arange(t*ll.dimxl, (t+1)*ll.dimxl)
                i_u = np.arange(t*ll.dimul, (t+1)*ll.dimul) + K * ll.dimxl
                xl_tp1, ul_t = X[i_x], X[i_u]
                J += (xl_tp1-ll.xd) @ ll.Ql2 @ (xl_tp1-ll.xd) + ul_t @ ll.Rl @ ul_t
            return J
        def con_dyn(X):
            f = np.zeros(K*ll.dimxl)
            for t in range(K):
                # get state and input component
                i_x = np.arange(t*ll.dimxl, (t+1)*ll.dimxl)
                i_u = np.arange(t*ll.dimul, (t+1)*ll.dimul) + K*ll.dimxl
                xl_tp1, ul_t = X[i_x], X[i_u]
                if t == 0:
                    xl_t = x0
                else:
                    xl_t = X[i_x-ll.dimxl]
                f[t*ll.dimxl: (t+1)*ll.dimxl] = xl_tp1 - ll.dyn(xl_t, ul_t)
            return f
        def con_safe(X):
            f = np.zeros(K*len(ll.obs))
            for t in range(K):
                i_x = np.arange(t*ll.dimxl, (t+1)*ll.dimxl)
                f[t*len(ll.obs): (t+1)*len(ll.obs)] = ll.safety_constr(X[i_x])
            return f
        def con_linear():  # 0 <= p <= ws_len, [vmin, wmin] <= u <= [vmax, wmax]
            a1 = np.array([[1,0,0,0.], [0,1,0,0.]])     # a1 @ x = p
            A1 = np.hstack( (np.kron(np.eye(K), a1), np.zeros((K*2,K*ll.dimul))) )
            lb1 = np.zeros(K*2) + 0.1               # robust margin
            ub1 = (ll.ws_len-0.1) * np.ones(K*2)    # robust margin
            
            A2 = np.hstack( (np.zeros((K*ll.dimul,K*ll.dimxl)), np.kron(np.eye(K), np.eye(ll.dimul))) )
            lb2 = np.kron(np.ones(K), np.array([ll.vmin, ll.wmin]))
            ub2 = np.kron(np.ones(K), np.array([ll.vmax, ll.wmax]))
            A = np.vstack( (A1,A2) )
            lb = np.concatenate( (lb1,lb2) )
            ub = np.concatenate( (ub1,ub2) )
            return LinearConstraint(A, lb, ub)      

        dxdy = param.dxdy
        grid_map = param.grid_map
        data = np.zeros( (N,K, 2*ll.dimxl+ll.dimul) )
        idx = []
        for i in range(N):
            print(f'Sample data point {i+1}/{N}...')
            # generate astar initial trajectory 
            x0 = ll.feas_x(xc=None, r=None)
            x_init = ll.astar_x_traj(x0, ll.xd, grid_map, dxdy, horizon=K+1)
            X0 = np.concatenate( (x_init[1:, :].flatten(), np.zeros(K*ll.dimul)) )
            constr = []
            constr.append( con_linear() )
            constr.append( NonlinearConstraint(con_dyn, np.zeros(K*ll.dimxl), np.zeros(K*ll.dimxl)) )
            constr.append( NonlinearConstraint(con_safe, np.zeros(K*len(ll.obs)), np.inf*np.ones(K*len(ll.obs))) )

            res = minimize(myobj, X0, constraints=constr, options={'maxiter': 100, 'disp': True})
            if res.status == 0:
                idx.append(i)
            else:
                continue
            
            x_traj = res.x[: K*ll.dimxl].reshape((K,ll.dimxl))
            u_traj = res.x[K*ll.dimxl: ].reshape((K,ll.dimul))
            
            # fill the data
            data[i, 0, : ll.dimxl] = x0
            data[i, 1: , : ll.dimxl] = x_traj[:-1, :]
            data[i,:, ll.dimxl: ] = np.hstack( (u_traj, x_traj) )
        
        return data[idx, :]


    def sample_follower_dyn_random(self, ff, N, K):
        '''
        Randomly sample follower's dynamical trajectory. 
        data[i, k, :] = [xf_k, uf_k, xf_kp1].
        '''
        data = np.zeros( (N,K, 2*ff.dimxf+ff.dimuf) )
        for i in range(N):
            print(f'Sample data point {i+1}/{N}...')
            x = ff.feas_x(xc=None, r=None)
            for k in range(K):
                u = ff.feas_u(x)
                x_new = ff.dyn(x, u)
                data[i, k, :] = np.concatenate( (x,u,x_new) )
                x = x_new
        return data
    

    def sample_follower_dynbr(self, ll, ff, N, K, data_l):
        '''
        Sample N trajectories of follower's K-step feedback dynamics using leader's trajectory.
        data[i, k, :] = [xf_k, ufopt_k, xf_kp1, xl_k, ul_k, xl_kp1].
        '''
        if N <= data_l.shape[0]:
            data_l = data_l[ll.rng.choice(data_l.shape[0], N, replace=False), :]
        else:
            raise Exception('N exceeds totoal number of leader\'s data samples.')
        
        data = np.zeros( (N,K, 2*ff.dimxf+ff.dimuf + 2*ll.dimxl+ll.dimul) )
        idx = []
        for i in range(N):
            print(f'Sample data point {i+1}/{N}...')
            xf = ff.feas_x(data_l[i, 0, : ll.dimxl], r=2)   # choose a nearby point as follower's initial state
            for k in range(K):
                print(f'  Sample stage {k}/{K}...')
                xl_k, ul_k = data_l[i, k, :ll.dimxl], data_l[i, k, ll.dimxl: ll.dimxl+ll.dimul]
                xl_kp1 = data_l[i, k, ll.dimxl+ll.dimul: ]
                
                uf_opt = ff.get_br(xf, xl_kp1)
                if uf_opt is None:
                    idx.append(i)   # record invalid trajectory index
                    break
                xf_new = ff.dyn(xf, uf_opt)
                data[i, k, :] = np.concatenate( (xf, uf_opt, xf_new, xl_k, ul_k, xl_kp1) )
                xf = xf_new
        
        # delete invalid data
        a = np.ones(N)
        a[idx] = 0
        data = data[np.nonzero(a)[0], :]
        return data
    
# delete later
    def sample_leader_dyn_opt_tmp1(self, ll, N, K, data_xl0):      # tmp, delete later
        '''
        Generate dynamic trajectory using astar path as initial condition. obj only u and xd.
        '''
        def myobj(X):   # X = [xl1, ..., xl_K, ul_0, ..., ul_Km1]
            J = 0
            for t in range(K):
                i_x = np.arange(t*ll.dimxl, (t+1)*ll.dimxl)
                i_u = np.arange(t*ll.dimul, (t+1)*ll.dimul) + K * ll.dimxl
                xl_tp1, ul_t = X[i_x], X[i_u]
                J += (xl_tp1-ll.xd) @ ll.Ql2 @ (xl_tp1-ll.xd) + ul_t @ ll.Rl @ ul_t
            return J
        def con_dyn(X):
            f = np.zeros(K*ll.dimxl)
            for t in range(K):
                # get state and input component
                i_x = np.arange(t*ll.dimxl, (t+1)*ll.dimxl)
                i_u = np.arange(t*ll.dimul, (t+1)*ll.dimul) + K*ll.dimxl
                xl_tp1, ul_t = X[i_x], X[i_u]
                if t == 0:
                    xl_t = x0
                else:
                    xl_t = X[i_x-ll.dimxl]
                f[t*ll.dimxl: (t+1)*ll.dimxl] = xl_tp1 - ll.dyn(xl_t, ul_t)
            return f
        def con_safe(X):
            f = np.zeros(K*len(ll.obs))
            for t in range(K):
                i_x = np.arange(t*ll.dimxl, (t+1)*ll.dimxl)
                f[t*len(ll.obs): (t+1)*len(ll.obs)] = ll.safety_constr(X[i_x])
            return f
        def con_linear():  # 0 <= p <= ws_len, [vmin, wmin] <= u <= [vmax, wmax]
            a1 = np.array([[1,0,0,0.], [0,1,0,0.]])     # a1 @ x = p
            A1 = np.hstack( (np.kron(np.eye(K), a1), np.zeros((K*2,K*ll.dimul))) )
            lb1 = np.zeros(K*2) + 0.1   # robust margin
            ub1 = (ll.ws_len-0.1) * np.ones(K*2)
            
            A2 = np.hstack( (np.zeros((K*ll.dimul,K*ll.dimxl)), np.kron(np.eye(K), np.eye(ll.dimul))) )
            lb2 = np.kron(np.ones(K), np.array([ll.vmin, ll.wmin]))
            ub2 = np.kron(np.ones(K), np.array([ll.vmax, ll.wmax]))
            A = np.vstack( (A1,A2) )
            lb = np.concatenate( (lb1,lb2) )
            ub = np.concatenate( (ub1,ub2) )
            return LinearConstraint(A, lb, ub)      

        dxdy = param.dxdy
        grid_map = param.grid_map
        data = np.zeros( (N,K, 2*ll.dimxl+ll.dimul) )
        idx = []
        for i in range(N):
            print(f'Sample data point {i+1}/{N}...')
            # generate astar initial trajectory 
            #x0 = ll.feas_x(xc=None, r=None)
            x0 = data_xl0[i, :]
            x_init = ll.astar_x_traj(x0, ll.xd, grid_map, dxdy, horizon=K+1)
            X0 = np.concatenate( (x_init[1:, :].flatten(), np.zeros(K*ll.dimul)) )
            constr = []
            constr.append( con_linear() )
            constr.append( NonlinearConstraint(con_dyn, np.zeros(K*ll.dimxl), np.zeros(K*ll.dimxl)) )
            constr.append( NonlinearConstraint(con_safe, np.zeros(K*len(ll.obs)), np.inf*np.ones(K*len(ll.obs))) )

            res = minimize(myobj, X0, constraints=constr, options={'maxiter': 100, 'disp': True})
            if res.status == 0:
                idx.append(i)
            else:
                continue
            
            x_traj = res.x[: K*ll.dimxl].reshape((K,ll.dimxl))
            u_traj = res.x[K*ll.dimxl: ].reshape((K,ll.dimul))
            
            # fill the data
            data[i, 0, : ll.dimxl] = x0
            data[i, 1: , : ll.dimxl] = x_traj[:-1, :]
            data[i,:, ll.dimxl: ] = np.hstack( (u_traj, x_traj) )
        
        return data[idx]


    def sample_follower_dynbr_tmp1(self, ll, ff, N, K, data_l=None, data_f=None):    # tmp delete later
        '''
        Sample N trajectories of follower's K-step feedback dynamics.
        random leader or astar leader or given leader's trajectory
        '''
        data = np.zeros( (N,K, 2*ff.dimxf+ff.dimuf + 2*ll.dimxl+ll.dimul) )

        for i in range(N):
            print(f'Sample data point {i+1}/{N}...')
            #xf = ff.feas_x(data_l[i, 0, : ll.dimxl], r=2)   # choose a nearby point as follower's initial state
            xf = data_f[i, :]   # use given initial data
            for k in range(K):
                print(f'  Sample stage {k}/{K}...')
                xl_k, ul_k = data_l[i, k, :ll.dimxl], data_l[i, k, ll.dimxl: ll.dimxl+ll.dimul]
                xl_kp1 = data_l[i, k, ll.dimxl+ll.dimul: ]
                
                uf_opt = ff.get_br(xf, xl_kp1)
                if uf_opt is None:
                    return data[:i, :]
                xf_new = ff.dyn(xf, uf_opt)
                data[i, k, :] = np.concatenate( (xf, uf_opt, xf_new, xl_k, ul_k, xl_kp1) )
                xf = xf_new
        
        return data
    

    def sample_leader_dyn_astar_old(self, ll, N, K):    # obsolete, keep name
        dxdy = param.dxdy
        grid_map = param.grid_map
        data = np.zeros( (N,K, 2*ll.dimxl+ll.dimul) )
        for i in range(N):
            print(f'Sample data point {i+1}/{N}...')
            x = ll.feas_x(xc=None, r=None)
            x_traj = ll.astar_x_traj(x, ll.xd, grid_map, dxdy, horizon=K)      # length K+1

            # compute u_traj
            for k in range(K):
                v = np.linalg.norm(x_traj[k+1, :-1] - x_traj[k, :-1]) / ll.dt
                v = np.min([v, ll.vmax])
                dtheta = (x_traj[k+1,-1] % (2*np.pi)) - (x_traj[k,-1] % (2*np.pi))
                if dtheta > np.pi:
                    dtheta = dtheta - 2*np.pi
                elif dtheta < -np.pi:
                    dtheta = 2*np.pi + dtheta
                else:
                    pass
                w = dtheta / ll.dt
                w = ll.wmin if w < ll.wmin else w
                w = ll.wmax if w > ll.wmax else w
                u = np.array([v,w])

                data[i, k, :] = np.concatenate( (x_traj[k,:], u, x_traj[k+1,:]) )
        return data



class PlotUtilities:
    def __init__(self) -> None:
        self.obs = param.obs_spec
        self.ws_len = param.ws_len


    def plot_grid_map(self, grid_map, dxdy=None):
        '''
        Plot grid map, black (obstacle), white (safe)
        coordinates: [x,y] = [i*dxdy, j*dxdy]
        '''
        a = np.array(grid_map)
        dxdy = 0.1
        x = np.arange(a.shape[0]) * dxdy
        y = np.arange(a.shape[1]) * dxdy
        fig, ax = plt.subplots()

        colors = [(1, 1, 1), (0, 0, 0)]  # White (0) to Black (1)
        n_bins = 2  # Increase this number for smoother color transitions'
        custom_cmap = mpl.colors.LinearSegmentedColormap.from_list('custom_cmap', colors, N=n_bins)
        ax.pcolormesh(x, y, a.T, cmap=custom_cmap, vmin=0, vmax=1)

        # plot grid
        ax.grid()
        ax.grid(which='minor', linestyle='--')
        ax.minorticks_on()

        ax.set_aspect(1)

        fig.savefig('tmp/tmp1.png', dpi=200)
        plt.close(fig)
        

    def plot_env(self, ax):
        '''
        This function plots the simulation environments.
        '''
        ax = self.plot_obs(ax)
        im = plt.imread('data/dest-logo.png')
        ax.imshow(im, extent=(8.4,9.6,8.4,9.6), zorder=-1)  # should be (8,9)
        
        ax.set_xlim(0, 10.1)
        ax.set_ylim(0, 10.1)
        ax.set_aspect(1)
        ax.xaxis.set_tick_params(labelsize='large')
        ax.yaxis.set_tick_params(labelsize='large')
        return ax


    def plot_obs(self, ax):
        '''
        This function plots the obstacles.
        '''
        def plot_l1_norm(ax, obs):      # plot l1 norm ball, a diamond
            xc, yc, rc = obs[0], obs[1], obs[2]
            x_scale, y_scale = obs[4], obs[5]
            # compute half width and height, construct vertex array
            width, height = rc*x_scale, rc*y_scale
            xy = [[xc, yc+height], [xc-width, yc], [xc, yc-height], [xc+width, yc]]
            p = mpatches.Polygon(np.array(xy), closed=True, ec='k', fc='0.9', fill=True, ls='--', lw=2)
            ax.add_patch(p)
            return ax
        
        def plot_l2_norm(ax, obs):      # plot l2 norm ball, an ellipse
            xc, yc, rc = obs[0], obs[1], obs[2]
            x_scale, y_scale = obs[4], obs[5]
            # compute width and height
            width, height = 2*rc*x_scale, 2*rc*y_scale
            p = mpatches.Ellipse((xc, yc), width, height, ec='k', fc='0.9', fill=True, ls='--', lw=2)
            ax.add_patch(p)
            return ax
        
        def plot_linf_norm(ax, obs):    # plot linf norm ball, a rectangle
            xc, yc, rc = obs[0], obs[1], obs[2]
            x_scale, y_scale = obs[4], obs[5]
            # compute x0, y0, width, and height
            width, height = 2*rc*x_scale, 2*rc*y_scale
            x0, y0 = xc-width/2, yc-height/2
            p = mpatches.Rectangle((x0,y0), width, height, ec='k', fc='0.9', fill=True, ls='--', lw=2)
            ax.add_patch(p)
            return ax            

        obs_num = len(self.obs)
        for i in range(obs_num):
            obs_i = self.obs[i]
            if obs_i[3] == 1:
                ax = plot_l1_norm(ax, obs_i)
            elif obs_i[3] == 2:
                ax = plot_l2_norm(ax, obs_i)
            else:
                ax = plot_linf_norm(ax, obs_i)
        
        return ax
    

    def plot_traj(self, xa_traj, xb_traj=None, c=None):
        '''
        This function plots the trajectory. First two dims are positions.
        '''
        fig, ax = plt.subplots()
        ax = self.plot_env(ax)
        if xb_traj is None:
            pa = xa_traj[:, 0:2]
            ax.plot(pa[:,0], pa[:,1], 'o', label='a')
        else:
            pa = xa_traj[:, 0:2]
            pb = xb_traj[:, 0:2]
            ax.plot(pa[:,0], pa[:,1], 'o', label='a')
            ax.plot(pb[:,0], pb[:,1], 's', label='b')
            #ax.plot(c[:,0], c[:,1], '^', label='c')
        ax.legend()
        fig.savefig('tmp/tmp3.png', dpi=200)
        plt.close(fig)
    

    def plot_loss(self, train_loss, test_loss=None):
        '''
        plot training loss and testing loss.
        '''
        fig, ax = plt.subplots()
        ax.plot(train_loss, label='train')
        if test_loss is not None:
            ax.plot(test_loss, label='test')
        ax.legend()
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        fig.savefig('tmp/tmp4.png', dpi=200)
        plt.close(fig)


    def plot_prederr(self, err_mean, err_std):
        '''
        Plot prediction error mean and variance
        '''
        fig, ax = plt.subplots()
        step = np.arange(err_mean.shape[0]) + 1
        ax.errorbar(step, err_mean, yerr=err_std, capsize=2, linestyle='--')
        ax.set_xticks(step)
        ax.set_xlabel('prediction step')
        ax.set_ylabel('error')
        fig.savefig('tmp/tmp5.png', dpi=200)
        plt.close(fig)