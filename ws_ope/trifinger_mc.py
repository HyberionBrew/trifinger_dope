import os
import sys
module_path = os.path.abspath(os.path.join('trifinger_rl_datasets'))
sys.path.insert(0, module_path)
module_path = os.path.abspath(os.path.join('/app/ws/trifinger-rl-example'))
sys.path.insert(0, module_path)
module_path = os.path.abspath(os.path.join('/app/ws/policy_eval'))
sys.path.insert(0, module_path)

import trifinger_rl_datasets
import gymnasium as gym
import numpy as np
import tensorflow as tf
import torch


from trifinger_rl_datasets import PolicyBase
from trifinger_rl_datasets.evaluate_sim import load_policy_class
import typing
import tqdm
from policy_eval.utils import TrifingerActor
from absl import flags
import pickle as pkl

FLAGS = flags.FLAGS

flags.DEFINE_boolean('normalize_states', True, 'Whether to normalize states.')
flags.DEFINE_boolean('normalize_rewards', True, 'Whether to normalize rewards.')
# number of mc_episodes to collect
flags.DEFINE_integer('num_mc_episodes', 50, 'Number of episodes to collect.')
flags.DEFINE_float('discount', 0.99, 'Discount used for returns.')
# since the behavior transformation uses the gpu we need this:
flags.DEFINE_string('trifinger_policy_class', "trifinger_rl_example.example.TorchPushPolicy",
                    'Policy class name for Trifinger.')
flags.DEFINE_float('target_policy_std', 0.1, 'Noise scale of target policy.')
flags.DEFINE_bool('target_policy_noisy', False, 'inject noise into the actions of the target policy')

# output file name
flags.DEFINE_string('output_file', 'mc_returns', 'Output file name.')

def run_episode(
        initial_obs: dict, initial_info: dict, actor: TrifingerActor, 
        env: gym.Env, discount: float, std: float = 0.05
    ):
    """Run one episode."""
    obs = initial_obs
    info = initial_info
    actor.reset() # reset the actor
    t = 0
    rewards = []
    episode_return = 0
    infos = []
    # print(actor.noisy)
    while True:
        _, action,_ = actor(obs, std=std)
        # print the action[0]
        # print(action[0])
        obs, rew, _, truncated, info = env.step(action[0])
        # if want to normalize the rewards
        # rew_norm = behavior_dataset.normalize_rewards([rew])[0] 
        infos.append(info)
        episode_return += rew * (discount**t)
        # rewards.append(rew_norm)
        t += 1
        if truncated: # should only be called @ t==750, actually it is at 748
            
            #if info["has_achieved"]:
            #    print("Success: Goal achieved at end of episode.")
            #else:
            #    print("Goal not reached at the end of the episode.")
            break
            
    return episode_return, None, infos

def estimate_monte_carlo_returns(env,
                                 discount,
                                 actor,
                                 std,
                                 num_episodes,
                                 max_length = 1000):
    """Estimate policy returns using with Monte Carlo."""
    # we reset after each episode
    episode_return_sum = 0
    eps = []
    infos = []
    _reset_time = 500
    pbar = tqdm.tqdm(range(num_episodes), desc='Running Monte Carlo')
    i = 0
    for i in pbar:

        # actually the env.reset() is only reset every 8 episodes 
        # in the original code, there is a soft reset otherwise (see below)

        if i%8 == 0:
            inital_obs, inital_info = env.reset()
        ep_return, rewards, info = run_episode(inital_obs, inital_info, actor, env, discount, std=std)
        # calculate discounted returns
        episode_return_sum += ep_return
        eps.append(ep_return)
        infos.append(info)
        # caluclate std of eps
        # pbar.set_postfix({'std:': np.std(eps)*(1 - discount)})
        pbar.set_postfix({'mean:': np.mean(eps)* (1 - discount)})
        # plot in this loop the rewards
        # plot the first 750 rewards
        # start soft-reset
        # move fingers to initial position and wait until cube has settled down
        env.reset_fingers(_reset_time) # as in original code
        if i < num_episodes - 1:
            # retrieve cube from barrier and center it approximately
            env.sim_env.reset_cube()
        # Sample new goal
        env.sim_env.sample_new_goal()
        initial_obs, initial_info = env.reset_fingers(_reset_time)
        # print("finished soft reset")

    return episode_return_sum / num_episodes * (1 - discount) , eps, infos

def count_successes(infos):
    successes = 0
    for i in range(len(infos)):
        if infos[i][len(infos[i])-1]["has_achieved"]==True:
            successes += 1
    return successes

def main(_):
    gpu_memory = 12209 # GPU memory available on the machine
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Restrict TensorFlow to only allocate 20% of the memory on the first GPU
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(gpu_memory * 0.2))])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e, " GPUs must be set at program startup")


    env_name = "trifinger-cube-push-sim-mixed-v0" # we just need the sim so it does not matter

    env = gym.make(
        "trifinger-cube-push-sim-mixed-v0",
        set_terminals =True,  # necessary since DOPE uses the terminals
        flatten_obs = True,   # obs as dictionary vs as array  
        image_obs = False,    # deactivate fetching images since large amounts of data
        visualization=False,  # enable visualization
    )
    dataset = env.get_dataset() # pull the dataset
    dataset.keys()
    
    # load the policy
    Policy = load_policy_class(FLAGS.policy_class)
    policy_config = Policy.get_policy_config()
    actor = Policy(env.action_space, env.observation_space, env.sim_env.episode_length)
    actor_wrap = TrifingerActor(actor, noisy= FLAGS.target_policy_noisy)
    ep_stats, rewards, infos = estimate_monte_carlo_returns(env, FLAGS.discount, actor_wrap,FLAGS.target_policy_std, FLAGS.num_mc_episodes)
    print(f"successes with low variance:  {count_successes(infos)}/{FLAGS.num_mc_episodes}")
    print("Mean return: ", ep_stats)
    with open(f"{FLAGS.output_file}.pkl", 'wb') as f:
        pkl.dump(rewards, f)
        pkl.dump(ep_stats, f)
        pkl.dump(infos, f)