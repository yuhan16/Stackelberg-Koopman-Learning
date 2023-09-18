"""
This module defines leader and follower classes.
"""
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint
import sg_koopman.parameters as param


class Agent:
    def __init__(self) -> None:
        self.rng = param.rng
        self.dt, self.dimT = param.dt, param.dimT
        self.obs = param.obs_spec
        self.ws_len = param.ws_len
    

    def safety_constr(self, x):
        '''
        This function generates the safety constraints c_j(x) = | Lam (p(x)-p^O_j) |_l - r_j >= 0, j=1,...,B.
        First two dims of x are positions
        '''
        p = x[0: 2]
        c = np.zeros(len(self.obs))

        for i in range(len(self.obs)):
            obs_i = self.obs[i]
            pxc, pyc, rc = obs_i[0], obs_i[1], obs_i[2]
            px_scale, py_scale = obs_i[4], obs_i[5] 
            f1 = np.diag([1/px_scale, 1/py_scale]) @ (p-np.array([pxc,pyc]))

            c[i] = np.linalg.norm(f1, ord=obs_i[3]) - rc - 0.1     # add safety margin
        
        return c
    
    
    def safety_constr_jac(self, x):
        '''
        This function computes the jacobian of the safety constraints c_j(x) = | Lam (p(x)-p^O_j) |_l - r_j >= 0, j=1,...,B.
        First two dims of x are positions
        '''
        p = x[0: 2]
        dc = np.zeros((len(self.obs), x.shape[0]))
            
        for i in range(len(self.obs)):
            obs_i = self.obs[i]
            pxc, pyc, rc = obs_i[0], obs_i[1], obs_i[2]
            px_scale, py_scale = obs_i[4], obs_i[5] 
            Lam = np.diag([1/px_scale, 1/py_scale])
            f1 = np.diag([1/px_scale, 1/py_scale]) @ (p-np.array([pxc,pyc]))

            if obs_i[3] == 1:
                dc[i, :2] = np.sign(f1) @ Lam
            elif obs_i[3] == 2:
                dc[i, :2] = f1 / np.linalg.norm(f1) @ Lam
            else:
                tmp = np.zeros_like(f1)
                idx = np.argmax(np.abs(f1))
                tmp[idx] = np.sign(f1[idx])
                dc[i, :2] = tmp @ Lam
        
        return dc
    

    def collision_detection(self, x, margin=0):
        '''
        This function detects if there is a collision at the given state x. Return True if a collision is detected.
        First two dims of x are positions. Consider magin for robust detection
        '''
        p = x[0: 2]

        if p[0] < 0 or p[0] > self.ws_len or p[1] < 0 or p[1] > self.ws_len:
            return True
        
        for i in range(len(self.obs)):
            obs_i = self.obs[i]
            pxc, pyc, rc = obs_i[0], obs_i[1], obs_i[2]
            px_scale, py_scale = obs_i[4], obs_i[5]
            
            f1 = np.diag([1/px_scale, 1/py_scale]) @ (p-np.array([pxc,pyc]))
            if np.linalg.norm(f1, ord=obs_i[3]) < rc + margin:  # consider safety margin for detection
                return True
        
        return False


    def target_detection(self, x):  # ??? still need
        dxdy = 0.25
        xd = param.xd
        if np.all( (x[0:2]/dxdy).astype(int) == (xd[0:2]/dxdy).astype(int) ):
            return True
        else:
            return False
        


class Node:
    """
    This class is designed for Astar algorithm.
    """
    def __init__(self, pos, parent=None) -> None:
        self.p = pos    # use list
        self.parent = parent
        self.f = 0
        self.g = 0
        self.h = 0
    
    
    def __eq__(self, node) -> bool:
        return self.p == node.p


    def dist(self, node):
        return abs(self.p[0]-node.p[0]) + abs(self.p[1]-node.p[1])


    def find_neighbors(self, maze):
        """
        maze is a 2D list, 0 for safe, 1 for obstacle, maze[x][y] is the xy position.
        """
        xmax, ymax = len(maze[0]), len(maze)
        candidate = [
            [self.p[0], self.p[1]+1], [self.p[0], self.p[1]-1], [self.p[0]+1, self.p[1]], [self.p[0]-1, self.p[1]], 
            [self.p[0]+1, self.p[1]+1], [self.p[0]+1, self.p[1]-1], [self.p[0]-1, self.p[1]+1], [self.p[0]-1, self.p[1]-1],
        ]

        node_nb = []
        for p in candidate:
            if p[0] < 0 or p[0] >= xmax:
                continue
            if p[1] < 0 or p[1] >= ymax:
                continue
            if maze[p[0]][p[1]] == 1:
                continue
            
            node_nb.append(Node(pos=p, parent=self))
        return node_nb



class Leader(Agent):
    def __init__(self) -> None:
        super().__init__()

        self.dimxl, self.dimul = param.dimxl, param.dimul
        self.xl0, self.xd = param.xl0, param.xd
        self.Ql1, self.Ql2, self.Rl = param.Ql1, param.Ql2, param.Rl
        self.Ql1_s2, self.Ql2_s2, self.Rl_s2 = param.Ql1_s2, param.Ql2_s2, param.Rl_s2
        
        self.dimxf, self.dimuf = param.dimxf, param.dimuf
        self.xf0 = param.xf0
        self.dimx = self.dimxl + self.dimxf

        self.vmin, self.vmax = param.vmin, param.vmax
        self.wmin, self.wmax = param.wmin, param.wmax
        
    
    def dyn1(self, x, u):
        '''
        Leader's dynamics, use heading angle.
        '''
        x_new = x + self.dt * np.array([ [np.cos(x[-1]), 0], [np.sin(x[-1]), 0], [0, 1]]) @ u   # unicycle dynamics
        return x_new
    

    def dyn_jac1(self, x, u):
        '''
        Jacobian df/dx and df/du of the leader's dynamics, using heading angle.
        '''
        dfdx = np.eye(self.dimxl) + self.dt * np.array([[0,0,-u[0]*np.sin(x[-1])], [0,0,u[0]*np.cos(x[-1])], [0,0,0]])
        dfdu = self.dt * np.array([[np.cos(x[-1]), 0], [np.sin(x[-1]), 0], [0, 1]])
        return dfdx, dfdu
    

    def dyn(self, x, u):
        '''
        This function implements the leader's dynamics. x = [px, py, cos_phi, sin_phi], u = [v, w]
        '''
        x_new = np.zeros_like(x)
        x_new[:2] = x[:2] + x[2:] * u[0]*self.dt
        x_new[2:] = np.array([ [x[2],-x[3]], [x[3],x[2]] ]) @ np.array([ np.cos(u[-1]*self.dt), np.sin(u[-1]*self.dt)]) 
        return x_new
    

    def dyn_jac(self, x, u):
        '''
        This function computes the jacobian df/dx and df/du of the leader's dynamics. x = [px, py, cos_phi, sin_phi], u = [v, w]
        '''
        dfdx = np.zeros((self.dimxf, self.dimxf))
        dfdx[:2, :2] = np.eye(2)
        dfdx[:2, 2:] = np.eye(2) * u[0]*self.dt
        dfdx[2:, 2:] = np.array([ [np.cos(u[-1]*self.dt), -np.sin(u[-1]*self.dt)], [np.sin(u[-1]*self.dt), np.cos(u[-1]*self.dt)] ])

        dfdu = np.zeros((self.dimxf, self.dimuf))
        dfdu[:2, 0] = np.array([x[2]*self.dt, x[3]*self.dt])
        dfdu[2:, 1] = np.array([-x[2]*np.sin(u[-1]*self.dt)*self.dt - x[3]*np.cos(u[-1]*self.dt)*self.dt, 
                                -x[3]*np.sin(u[-1]*self.dt)*self.dt + x[2]*np.cos(u[-1]*self.dt)*self.dt])
        return dfdx, dfdu
    

    def feas_x(self, xc=None, r=None):
        '''
        Sample a feasible state x centered around xc with radius r. Dynamics related, direction (phi angle) is not affected.
        '''
        while True:
            if xc is None or r is None:
                p = self.rng.uniform(0, self.ws_len, 2)
                theta = self.rng.uniform(0, 2*np.pi)
                d = np.array([np.cos(theta), np.sin(theta)])
                x = np.concatenate( (p,d) )
            else:
                p = self.rng.uniform(-1, 1, 2) * 2 + xc[:2]
                theta = self.rng.uniform(0, 2*np.pi)
                d = np.array([np.cos(theta), np.sin(theta)])
                x = np.concatenate( (p,d) )
            
            if not self.collision_detection(x, margin=0.1):
                return x
    

    def feas_u(self, x):
        '''
        Sample one-step feasible control u based on the dynamics, starting from x. Dynamics related.
        u = [v,w], vmin <= v <= vmax, wmin <= w <= wmax.
        '''
        p, d = x[:2], x[2:]
        
        # linear search in the heading direction
        inc = 0.1
        vhigh = 0
        for scale in np.arange(0, 1+inc, inc):
            p_new = p + self.dt * d * (scale*self.vmax)
            if self.collision_detection(p_new, margin=0.1):
                break
            vhigh = scale*self.vmax        # need to record after if condition
        
        v = self.rng.uniform(low=self.vmin, high=vhigh)
        w = self.rng.uniform(low=self.wmin, high=self.wmax)
        
        return np.array([v,w])
        
        
    def astar(self, maze, p0, pf):
        '''
        This function implements A star algorithm given a grid map.
        p0, pf are cell index, not positions. Data type use list
        '''
        open_list = []
        close_list = []
        start = Node(p0)
        target = Node(pf)
        open_list.append(start)

        while len(open_list) > 0:
            # 1. find node with least f
            q = open_list[0]
            for node in open_list:
                if q.f > node.f:
                    q = node

            # 2.pop q in open_list and append to the closed_list
            open_list.remove(q)
            close_list.append(q)

            # 3. check if q is target
            if q == target:
                path = []
                item = close_list[-1]
                while item.parent is not None:
                    path.append(item.p)
                    item = item.parent
                path.append(item.p)     # append the last node
                path.reverse()
                return path
        
            # 4. find neighbors of q
            node_nb = q.find_neighbors(maze)
            for node in node_nb:
                # 4a. check node in closed list, if so, jump this node
                flag = False
                for item in close_list:
                    if item == node:
                        flag = True
                        break
                if flag:
                    continue
            
                # 4b. compute g, h, f for neighbors
                node.g = q.g + 1
                node.h = node.dist(target)
                node.f = node.g + node.h

                # 4c. check node in open_list, update node if new g is smaller, or append if not in the list
                flag = False
                for item in open_list:
                    if item == node: 
                        flag = True
                        if node.g < item.g:
                            item.g, item.f = node.g, node.f
                        break
                if not flag:
                    open_list.append(node)


    def astar_p_traj(self, path, dxdy):
        '''
        Return a position trajectory using path found by Astar algorithm.
        - path[0] and path[-1] are starting and end cells.
        '''
        traj = []
        for i in range(len(path)):
            traj.append( np.array(path[i])*dxdy + np.ones(2)*dxdy/2 )   # use cell center position
        
        return np.array(traj) 
    

    def astar_phi_traj(self, p_traj):
        '''
        Return a direction trajectory found by Astar algorithm. 
        phi[-1] is always set by 0. phi[k] is determined by p[k] and p[k+1].
        '''
        phi_traj = np.zeros((p_traj.shape[0], 1))
        for t in range(p_traj.shape[0]-1):
            phi_traj[t, 0] = np.arctan2(p_traj[t+1,1]-p_traj[t,1], p_traj[t+1,0]-p_traj[t,0])
        d_traj = np.hstack( (np.cos(phi_traj), np.sin(phi_traj)) )
        return d_traj


    def astar_x_traj(self, x0, xd, grid_map, dxdy, horizon=None):
        '''
        Generate a trajectory using a_star. x0 can be either leader or follower state. 
        '''
        def get_corners(p):
            return [p+np.array([0,1]), p-np.array([0,1]), p+np.array([1,0]), p-np.array([1,0]), 
                    p+np.array([1,1]), p-np.array([1,1]), p+np.array([1,-1]), p-np.array([1,-1])]
        
        # get coordinate in the grid map
        p0 = (x0[0:2]/dxdy).astype(int)
        pd = (xd[0:2]/dxdy).astype(int)
        
        # x0,xd (p0,pd) may be in the obstacle cell due to the grid_map resolution, find the nearest corners
        if grid_map[p0[0]][p0[1]] == 1:
            for c in get_corners(p0):
                if grid_map[c[0]][c[1]] == 0:
                    p0 = c
                    break
        if grid_map[pd[0]][pd[1]] == 1:
            for c in get_corners(pd):
                if grid_map[c[0]][c[1]] == 0:
                    pd = c
                    break
        
        p_path = self.astar(grid_map, p0.tolist(), pd.tolist())
        p_traj = self.astar_p_traj(p_path, dxdy)
        phi_traj = self.astar_phi_traj(p_traj)
        x_traj = np.hstack( (p_traj, phi_traj) )
    
        if horizon is None:
            horizon = x_traj.shape[0]

        # add trajectory to full
        if x_traj.shape[0] < horizon:
            tmp = np.ones((horizon-x_traj.shape[0],1)) @ x_traj[[-1],:]
            x_traj = np.vstack((x_traj, tmp))
        else:
            x_traj = x_traj[: horizon, :]
        x_traj[0, :] = x0   # x_traj[0, :] is cell center not x0, replace with x0

        return x_traj
        


class Follower(Agent):
    def __init__(self) -> None:
        super().__init__()

        self.dimxf, self.dimuf = param.dimxf, param.dimuf
        self.xf0, self.xd = param.xf0, param.xd
        self.Qf1, self.Qf2, self.Qf3, self.Rf = param.Qf1, param.Qf2, param.Qf3, param.Rf

        self.vmin, self.vmax = param.vmin, param.vmax
        self.wmin, self.wmax = param.wmin, param.wmax


    def dyn1(self, x, u):
        """
        This function implements the follower's dynamics, use heading angle.
        x = [px, py, phi], u = [v, w]
        """
        x_new = x + self.dt * np.array([ [np.cos(x[-1]), 0], [np.sin(x[-1]), 0], [0, 1] ]) @ u
        return x_new


    def dyn_jac1(self, x, u):
        """
        Jacobian df/dx and df/du of the follower's dynamics, use heading angle. 
        x = [px, py, phi], u = [v, w]
        """
        dfdx = np.eye(self.dimxf) + self.dt * np.array([[0,0,-u[0]*np.sin(x[-1])], [0,0,u[0]*np.cos(x[-1])], [0,0,0]])
        dfdu = self.dt * np.array([[np.cos(x[-1]), 0], [np.sin(x[-1]), 0], [0, 1]])
        return dfdx, dfdu
    

    def dyn(self, x, u):
        '''
        Follower's dynamics. x = [px, py, cos_phi, sin_phi], u = [v, w]
        '''
        x_new = np.zeros_like(x)
        x_new[:2] = x[:2] + x[2:] * u[0]*self.dt
        x_new[2:] = np.array([ [x[2],-x[3]], [x[3],x[2]] ]) @ np.array([ np.cos(u[-1]*self.dt), np.sin(u[-1]*self.dt)]) 
        return x_new
    

    def dyn_jac(self, x, u):
        '''
        Jacobian df/dx and df/du of the follower's dynamics. x = [px, py, cos_phi, sin_phi], u = [v, w]
        '''
        dfdx = np.zeros((self.dimxf, self.dimxf))
        dfdx[:2, :2] = np.eye(2)
        dfdx[:2, 2:] = np.eye(2) * u[0]*self.dt
        dfdx[2:, 2:] = np.array([ [np.cos(u[-1]*self.dt), -np.sin(u[-1]*self.dt)], [np.sin(u[-1]*self.dt), np.cos(u[-1]*self.dt)] ])

        dfdu = np.zeros((self.dimxf, self.dimuf))
        dfdu[:2, 0] = np.array([x[2]*self.dt, x[3]*self.dt])
        dfdu[2:, 1] = np.array([-x[2]*np.sin(u[-1]*self.dt)*self.dt - x[3]*np.cos(u[-1]*self.dt)*self.dt, 
                                -x[3]*np.sin(u[-1]*self.dt)*self.dt + x[2]*np.cos(u[-1]*self.dt)*self.dt])
        return dfdx, dfdu


    def feas_x(self, xc=None, r=None):
        '''
        Sample a feasible state x centered around xc with radius r. Dynamics related, phi is not affected.
        '''
        while True:
            if xc is None or r is None:
                p = self.rng.uniform(0, self.ws_len, 2)
                theta = self.rng.uniform(0, 2*np.pi)
                d = np.array([np.cos(theta), np.sin(theta)])
                x = np.concatenate( (p,d) )
            else:
                p = self.rng.uniform(-1, 1, 2) * r + xc[:2]
                theta = self.rng.uniform(0, 2*np.pi)
                d = np.array([np.cos(theta), np.sin(theta)])
                x = np.concatenate( (p,d) )

            if not self.collision_detection(x, margin=0.1):
                return x
    

    def feas_u(self, x):
        '''
        Sample one-step feasible control u based on the dynamics, starting from x. Dynamics related.
        u = [v,w], vmin <= v <= vmax, wmin <= w <= wmax.
        '''
        p, d = x[:2], x[2:]
        
        # linear search in the heading direction
        inc = 0.1
        vhigh = 0
        for scale in np.arange(0, 1+inc, inc):
            p_new = p + self.dt * d * (scale*self.vmax)
            if self.collision_detection(p_new, margin=0.1):
                break
            vhigh = scale*self.vmax        # need to record after if condition
        
        v = self.rng.uniform(low=self.vmin, high=vhigh)
        w = self.rng.uniform(low=self.wmin, high=self.wmax)
        
        return np.array([v,w])


    def get_br(self, xf, xl_new):
        '''
        This function computes the follower's optimal response given the current state xf and leader's new state xl_new.
        Passing xl_new to the follower simplifies the computation.
        '''
        constr = []
        dl_new = xl_new[2:]

        def myobj(X):   # X = [xf_new, uf]
            x, u = X[ :self.dimxf], X[self.dimxf: ]
            J = (x-xl_new) @ self.Qf1 @ (x-xl_new) + (x-self.xd) @ self.Qf2 @ (x-self.xd) + self.Qf3 * dl_new @ x[2:] + u @ self.Rf @ u 
            return J
        def myobj_jac(X):
            x, u = X[ :self.dimxf], X[self.dimxf: ]
            jac = np.zeros_like(X)
            jac[: self.dimxf] = 2*self.Qf1 @ (x-xl_new) + 2*self.Qf2 @ (x-self.xd)
            jac[2: self.dimxf] = self.Qf3 * dl_new
            jac[self.dimxf: ] = 2*self.Rf @ u
            return jac

        def dyn_constr(X):
            x, u = X[ :self.dimxf], X[self.dimxf: ]
            c = x - self.dyn(xf, u)
            return c
        def dyn_constr_jac(X):
            x, u = X[ :self.dimxf], X[self.dimxf: ]
            _, dfdu = self.dyn_jac(xf, u)
            dc = np.zeros((self.dimxf, self.dimxf+self.dimuf))
            dc[:, : self.dimxf] = np.eye(self.dimxf)
            dc[:, self.dimxf: ] = -dfdu
            return dc
        constr.append(NonlinearConstraint(dyn_constr, np.zeros(self.dimxf), np.zeros(self.dimxf), jac=dyn_constr_jac))

        def safe_constr(X):
            x, u = X[ :self.dimxf], X[self.dimxf: ]
            c = self.safety_constr(x)   # c(x) >= 0
            return c
        def safe_constr_jac(X):
            dc = self.safety_constr_jac(X)  # pass X, return correct jacobian size
            return dc
        constr.append(NonlinearConstraint(safe_constr, np.zeros(len(self.obs)), np.inf*np.ones(len(self.obs)), jac=safe_constr_jac))

        # input constr, [vmin, wmin] <= [v, w] <= [vmax, wmax]
        AA = np.hstack( (np.zeros((self.dimuf, self.dimxf)), np.eye(self.dimuf)) )
        constr.append( LinearConstraint(AA, lb=np.array([self.vmin, self.wmin]), ub=np.array([self.vmax, self.wmax])) )

        # range constr, [0, 0] <= [px, py] <= [ws_len, ws_len]
        BB = np.hstack( (np.eye(2), np.zeros((2,2+self.dimuf))) )
        constr.append( LinearConstraint(BB, lb=np.zeros(2)+0.1, ub=self.ws_len*np.ones(2)-0.1) )    # add robust margin

        '''
        #---------- test jac -----------#
        from scipy.optimize import approx_fprime, check_grad
        tmp = np.zeros(50)
        for ii in range(tmp.shape[0]):
            X00 = self.rng.random(self.dimxf+self.dimuf) * 5 - 1
            tmp[ii] = check_grad(myobj, myobj_jac, X00)
            #tmp[ii] = check_grad(dyn_constr, dyn_constr_jac, X00)
            #tmp[ii] = check_grad(safe_constr, safe_constr_jac, X00)
        print(tmp)
        #-------------------------------#
        '''

        # grid search for global optimal solution
        inc_v = 0.2
        inc_w = 0.2
        Xinit = []
        for v in np.arange(self.vmin, self.vmax+inc_v, inc_v):
            for w in np.arange(self.wmin, self.wmax+inc_w, inc_w):
                uf = np.array([v, w])
                xf_new = self.dyn(xf, uf)
                if not self.collision_detection(xf_new):
                    Xinit.append( np.concatenate((xf_new+0.1*self.rng.random(self.dimxf),uf)) )  # add some random noise
        
        if len(Xinit) == 0:
            #print(xf, self.collision_detection(xf))
            #print(self.safety_constr(xf))
            print('Empty initial conditions. No feasible action. Return None.')
            return None
        
        fval = np.zeros(len(Xinit))
        uf_opt = np.zeros((len(Xinit), self.dimuf))
        for i in range(len(Xinit)):
            X0 = Xinit[i]
            res = minimize(myobj, X0, jac=myobj_jac, constraints=constr, options={'maxiter': 100, 'disp': False})
            print('status {}: {}.'.format(res.status, res.message))
            if res.status == 0:
                fval[i] = res.fun
                uf_opt[i, :] = res.x[self.dimxf: ]
            else:
                fval[i] = np.inf
        ii = np.argmin(np.array(fval))
        print(f'  valid X0: {len(Xinit)}, fopt: {fval[ii]:.4f}, uopt: {uf_opt[ii]}, xf: {xf}, xl_new: {xl_new}.')

        return uf_opt[ii]


    def no_guide_br(self, xf):
        '''
        Compute next move when no guidance cost. heading direction align with theta=0.
        '''
        constr = []
        target_d = np.array([np.cos(0), np.sin(0)])

        def myobj(X):   # X = [xf_new, uf]
            x, u = X[ :self.dimxf], X[self.dimxf: ]
            J = (x-self.xd) @ self.Qf2 @ (x-self.xd) + self.Qf3 * target_d @ x[2:] + u @ self.Rf @ u 
            return J
        def myobj_jac(X):
            x, u = X[ :self.dimxf], X[self.dimxf: ]
            jac = np.zeros_like(X)
            jac[: self.dimxf] = 2*self.Qf2 @ (x-self.xd)
            jac[2: self.dimxf] = self.Qf3 * target_d
            jac[self.dimxf: ] = 2*self.Rf @ u
            return jac

        def dyn_constr(X):
            x, u = X[ :self.dimxf], X[self.dimxf: ]
            c = x - self.dyn(xf, u)
            return c
        def dyn_constr_jac(X):
            x, u = X[ :self.dimxf], X[self.dimxf: ]
            _, dfdu = self.dyn_jac(xf, u)
            dc = np.zeros((self.dimxf, self.dimxf+self.dimuf))
            dc[:, : self.dimxf] = np.eye(self.dimxf)
            dc[:, self.dimxf: ] = -dfdu
            return dc
        constr.append(NonlinearConstraint(dyn_constr, np.zeros(self.dimxf), np.zeros(self.dimxf), jac=dyn_constr_jac))

        def safe_constr(X):
            x, u = X[ :self.dimxf], X[self.dimxf: ]
            c = self.safety_constr(x)   # c(x) >= 0
            return c
        def safe_constr_jac(X):
            dc = self.safety_constr_jac(X)  # pass X, return correct jacobian size
            return dc
        constr.append(NonlinearConstraint(safe_constr, np.zeros(len(self.obs)), np.inf*np.ones(len(self.obs)), jac=safe_constr_jac))

        # input constr, [vmin, wmin] <= [v, w] <= [vmax, wmax]
        AA = np.hstack( (np.zeros((self.dimuf, self.dimxf)), np.eye(self.dimuf)) )
        constr.append( LinearConstraint(AA, lb=np.array([self.vmin, self.wmin]), ub=np.array([self.vmax, self.wmax])) )

        # range constr, [0, 0] <= [px, py] <= [ws_len, ws_len]
        BB = np.hstack( (np.eye(2), np.zeros((2,2+self.dimuf))) )
        constr.append( LinearConstraint(BB, lb=np.zeros(2)+0.1, ub=self.ws_len*np.ones(2)-0.1) )    # add robust margin

        '''
        #---------- test jac -----------#
        from scipy.optimize import approx_fprime, check_grad
        tmp = np.zeros(50)
        for ii in range(tmp.shape[0]):
            X00 = self.rng.random(self.dimxf+self.dimuf) * 5 - 1
            tmp[ii] = check_grad(myobj, myobj_jac, X00)
            #tmp[ii] = check_grad(dyn_constr, dyn_constr_jac, X00)
            #tmp[ii] = check_grad(safe_constr, safe_constr_jac, X00)
        print(tmp)
        #-------------------------------#
        '''

        # grid search for global optimal solution
        inc_v = 0.2
        inc_w = 0.2
        Xinit = []
        for v in np.arange(self.vmin, self.vmax+inc_v, inc_v):
            for w in np.arange(self.wmin, self.wmax+inc_w, inc_w):
                uf = np.array([v, w])
                xf_new = self.dyn(xf, uf)
                if not self.collision_detection(xf_new):
                    Xinit.append( np.concatenate((xf_new+0.1*self.rng.random(self.dimxf),uf)) )  # add some random noise
        
        if len(Xinit) == 0:
            #print(xf, self.collision_detection(xf))
            #print(self.safety_constr(xf))
            print('Empty initial conditions. No feasible action. Return None.')
            return None
            
        fval = np.zeros(len(Xinit))
        uf_opt = np.zeros((len(Xinit), self.dimuf))
        for i in range(len(Xinit)):
            X0 = Xinit[i]
            res = minimize(myobj, X0, jac=myobj_jac, constraints=constr, options={'maxiter': 100, 'disp': False})
            print('status {}: {}.'.format(res.status, res.message))
            if res.status == 0:
                fval[i] = res.fun
                uf_opt[i, :] = res.x[self.dimxf: ]
            else:
                fval[i] = np.inf
        ii = np.argmin(np.array(fval))
        print(f'  valid X0: {len(Xinit)}, fopt: {fval[ii]:.4f}, uopt: {uf_opt[ii]}, xf: {xf}.')

        return uf_opt[ii]
        