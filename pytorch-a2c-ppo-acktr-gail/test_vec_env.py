import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from tqdm import tqdm_notebook as tqdm
import numpy as np
import torch as T
import torch.nn as nn
import sys
from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize
import pdb 

sys.path.append('a2c_ppo_acktr')

env_names = ['HalfCheetah-v2', 'Hopper-v2', 'Ant-v2', 'Reacher-v2']
env_id = 0
env_name = env_names[env_id]
seed = 1

pdb.set_trace()
env = make_vec_envs(
    env_name,
    seed + 1000,
    1,
    None,
    None,
    device='cpu',
    allow_early_resets=False)

pdb.set_trace()
obs = env.reset()
st = env.venv.envs[0].sim.get_state()
print(obs)
print(st)
act = np.random.normal(demo_A_mean, demo_A_std, dU).astype(np.float32)
new_obs, r, done, _ = env.step(T.from_numpy(np.array([act])))
new_obs = new_obs.numpy()[0]
st = env.venv.envs[0].sim.get_state()
print(new_obs)
print(st)
