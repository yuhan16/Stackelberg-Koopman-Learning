'''
Generate training and testing data for all scenarios.
'''
import sys, os
sys.path.insert(1, os.path.realpath('.'))

from sg_koopman.common.utils import SamplingUtilities
from sg_koopman.common.agents import Leader, Follower
import numpy as np


# create data directories for different algorithms
if not os.path.exists('data/kp_nn'):
    os.mkdir('data/kp_nn')

if not os.path.exists('data/nn_fdynbr'):
    os.mkdir('data/nn_fdynbr')

if not os.path.exists('data/nonlin_ocp'):
    os.mkdir('data/nonlin_ocp')

if not os.path.exists('data/dmd'):
    os.mkdir('data/dmd')


def generate_dyn_data():
    ll = Leader()
    ff = Follower()
    util = SamplingUtilities()
    N = 10000
    K = 10

    # generate data for leader's dynamics
    print(f'Generating {N} trajectories (length {K}) of leader\'s dynamics using random control...')
    Dl_random = util.sample_leader_dyn_random(ll, N=N, K=K)
    np.save('leader_dyn_rand.npy', Dl_random)
    print('Done.\n')

    print(f'Generating {N} trajectories (length {K}) of leader\'s dynamics using Astar planning...')
    Dl_astar = util.sample_leader_dyn_astar(ll, N=N, K=K)
    np.save('leader_dyn_astar.npy', Dl_astar)
    print('Done.\n')


    # generate data for follower
    print(f'Generate {N} trajectories (length {K}) of follower\'s dynamics using random control...')
    Df_random = util.sample_follower_dyn_random(ff, N=N, K=K)
    np.save('follower_dyn_rand.npy', Df_random)
    print('Done.\n')
    

def generate_fdynbr_data():
    '''
    Generate follower's feedback dynamics data using existing leader's trajectory.
    '''
    # prepare leader's trajectory, use random or astar or mix
    if os.path.exists('data/leader_dyn_rand.npy') and os.path.exists('data/leader_dyn_astar.npy'):
        Dl_random = np.load('data/leader_dyn_rand.npy')
        Dl_astar = np.load('data/leader_dyn_astar.npy')
        Dl_mix = np.concatenate((Dl_random, Dl_astar), axis=0)
    else:
        raise Exception('No leader dyn data found. Run generate_dyn_data() first.')

    ll = Leader()
    ff = Follower()
    util = SamplingUtilities()
    N = min(10000, Dl_mix.shape[0])
    K = 30
    print(f'Generate {N} trajectories (length {K}) of follower\'s feedback dynamics...')
    Dfb_traj = util.sample_follower_dynbr(ll, ff, N=N, K=K, data_l=Dl_mix)   # or use Dl_random, Dl_astar
    np.save('follower_dynbr_lmix.npy', Dfb_traj)    # lmix means using Dl_mix
    print('Done.\n')


def generate_dyn_multiproc():
    '''
    Use multiprocessing to accelerate dyn data generation.
    '''
    import multiprocessing, time
    util = SamplingUtilities()
    n_worker = 20
    N = 10000
    K = 30

    # multi processing
    arg_list = []
    N_batch = N // n_worker
    for i in range(n_worker):     # num of parallel tasks
        leader = Leader()
        leader.rng = np.random.default_rng(np.random.randint(100000000))   # use different seed
        #arg_list.append( (leader, N_batch, K) )

        follower = Follower()
        follower.rng = np.random.default_rng(np.random.randint(100000000))  # use different seed
        arg_list.append( (follower, N_batch, K) )
    
    with multiprocessing.Pool() as pool:
        st = time.time()
        #res_mp = pool.starmap(util.sample_leader_dyn_random, arg_list)
        #res_mp = pool.starmap(util.sample_leader_dyn_astar, arg_list)
        res_mp = pool.starmap(util.sample_follower_dyn_random, arg_list)
        elapsed_time = time.time() - st
    print("total time for multi-processing: {:.3f} min".format(elapsed_time/60))
    
    print(np.concatenate(res_mp, axis=0).shape)
    #np.save('leader_dyn_rand.npy', np.concatenate(res_mp, axis=0))
    #np.save('leader_dyn_astar.npy', np.concatenate(res_mp, axis=0))
    np.save('follower_dyn_rand.npy', np.concatenate(res_mp, axis=0))


def generate_fdynbr_multiproc():
    '''
    Use multiprocessing to accelerate fdynbr data generation.
    '''
    # prepare leader's trajectory, use random or astar or mix
    if os.path.exists('data/leader_dyn_rand.npy') and os.path.exists('data/leader_dyn_astar.npy'):
        Dl_random = np.load('data/leader_dyn_rand.npy')
        Dl_astar = np.load('data/leader_dyn_astar.npy')
        Dl_mix = np.concatenate((Dl_random, Dl_astar), axis=0)
    else:
        raise Exception('No leader dyn data found. Run generate_dyn_data() first.')
    
    import multiprocessing, time
    util = SamplingUtilities()
    N = min(10000, Dl_mix.shape[0])
    K = 30
    n_worker = 20

    arg_list = []
    N_batch = N // n_worker
    for i in range(n_worker):     # num of parallel tasks
        leader = Leader()
        leader.rng = np.random.default_rng(np.random.randint(100000000))
        follower = Follower()
        follower.rng = np.random.default_rng(np.random.randint(100000000))
        arg_list.append( (leader, follower, N_batch, K, Dl_mix[i*N_batch: (i+1)*N_batch]) )
    
    with multiprocessing.Pool() as pool:
        st = time.time()
        res_mp = pool.starmap(util.sample_follower_dynbr, arg_list)
        elapsed_time = time.time() - st
    print("total time for multi-processing: {:.3f} min".format(elapsed_time/60))

    print(np.concatenate(res_mp, axis=0).shape)
    np.save('follower_dynbr_lmix.npy', np.concatenate(res_mp, axis=0))
    


if __name__ == '__main__':
    generate_dyn_data()
    generate_fdynbr_data()

    #generate_dyn_multiproc()
    #generate_fdynbr_multiproc()