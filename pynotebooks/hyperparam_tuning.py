#!/usr/bin/env python
# coding: utf-8

import numpy as np
import keras
import tensorflow
import gym

from Function_Library import *
from Environments import *

import rl as rl
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy, GreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger

import json
import copy
import sys
import os
import shutil
import datetime
import pickle

LR = [10*10**-5, 1*10**-5]
N = [100, 50, 10]
hypergrid = []
for rate in LR:
    for nstep in N:
        hypergrid.append({"rate":rate, "nstep":nstep})

for run_index, grid in enumerate(hypergrid):

    fixed_configs = {"d": 5,
                    "use_Y": False,
                    "train_freq": 1,
                    "batch_size": 32,
                    "print_freq": 250,
                    "rolling_average_length": 500,
                    "stopping_patience": 500,
                    "error_model": "X",
                    "c_layers": [[64,3,2],[32,2,1],[32,2,1]],
                    "ff_layers": [[512,0.2]],
                    "max_timesteps": 100000,
                    "volume_depth": 5,
                    "testing_length": 101,
                    "buffer_size": 50000,
                    "dueling": True,
                    "masked_greedy": False,
                    "static_decoder": True}

    variable_configs = {"p_phys": 0.013,
                        "p_meas": 0.013,
                        "success_threshold": 70,
                        "learning_starts": 1000,
                        "learning_rate": grid["rate"],
                        "exploration_fraction": 100000,
                        "max_eps": 0.5,
                        "target_network_update_freq": 5000,
                        "gamma": 0.99,
                        "final_eps": 0.001}

    logging_directory = os.path.join(os.getcwd(),"logging_directory/" + str(run_index+1) + "/")
    static_decoder_path = os.path.join(os.getcwd(),"referee_decoders/nn_d5_X_p5")


    all_configs = {}

    for key in fixed_configs.keys():
        all_configs[key] = fixed_configs[key]

    for key in variable_configs.keys():
        all_configs[key] = variable_configs[key]

    static_decoder = load_model(static_decoder_path)                                                 
    logging_path = os.path.join(logging_directory,"training_history.json")
    logging_callback = FileLogger(filepath = logging_path,interval = all_configs["print_freq"])

    env = Surface_Code_Environment_Multi_Decoding_Cycles(d=all_configs["d"], 
        p_phys=all_configs["p_phys"], 
        p_meas=all_configs["p_meas"],  
        error_model=all_configs["error_model"], 
        use_Y=all_configs["use_Y"], 
        volume_depth=all_configs["volume_depth"],
        static_decoder=static_decoder)

    memory = SequentialMemory(limit=all_configs["buffer_size"], nsteps=grid["nstep"], window_length=1)

    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(masked_greedy=all_configs["masked_greedy"]), 
        attr='eps', value_max=all_configs["max_eps"], 
        value_min=all_configs["final_eps"], 
        value_test=0.0, 
        nb_steps=all_configs["exploration_fraction"])

    test_policy = GreedyQPolicy(masked_greedy=True)

    model = build_convolutional_nn(all_configs["c_layers"], 
                                   all_configs["ff_layers"], 
                                   env.observation_space.shape, 
                                   env.num_actions)

    dqn = DQNAgent(model=model, 
                   nb_actions=env.num_actions, 
                   memory=memory, 
                   nb_steps_warmup=all_configs["learning_starts"], 
                   target_model_update=all_configs["target_network_update_freq"], 
                   policy=policy,
                   test_policy = test_policy,
                   gamma = all_configs["gamma"],
                   enable_dueling_network=all_configs["dueling"])  


    dqn.compile(Adam(lr=all_configs["learning_rate"]))


    now = datetime.datetime.now()
    started_file = os.path.join(logging_directory,"started_at.p")
    pickle.dump(now, open(started_file, "wb" ) )

    history = dqn.fit(env, 
      nb_steps=all_configs["max_timesteps"], 
      action_repetition=1, 
      callbacks=[logging_callback], 
      verbose=2,
      visualize=False, 
      nb_max_start_steps=0, 
      start_step_policy=None, 
      log_interval=all_configs["print_freq"],
      nb_max_episode_steps=None, 
      episode_averaging_length=all_configs["rolling_average_length"], 
      success_threshold=all_configs["success_threshold"],
      stopping_patience=all_configs["stopping_patience"],
      min_nb_steps=all_configs["exploration_fraction"],
      single_cycle=False)


    weights_file = os.path.join(logging_directory, "dqn_weights.h5f")
    dqn.save_weights(weights_file, overwrite=True)
    
    historypickle = os.path.join(logging_directory,"historypickle.p")
    pickle.dump(history, open(historypickle, "wb" ) )

# And finally, in order to evaluate the training procedure we may be interested in viewing any of the metrics which were logged. These are all saved within the history.history dictionary. For example, we are often most interested in analyzing the training procedure by looking at the rolling average of the qubit lifetime, which we can do as follows:

# In[10]:


# from matplotlib import pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

# training_history = history.history["episode_lifetimes_rolling_avg"]

# plt.figure(figsize=(12,7))
# plt.plot(training_history)
# plt.xlabel('Episode')
# plt.ylabel('Rolling Average Qubit Lifetime')
# _ = plt.title("Training History")


# # From the above plot one can see that during the exploration phase the agent was unable to do well, due to constant exploratory random actions, but was able to exploit this knowledge effectively once the exploration probability became sufficiently low. Again, it is also clear that the agent was definitely still learning and improving when we chose to stop the training procedure.

# # In[8]:


# from matplotlib import pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

# training_history = history.history["episode_lifetimes_rolling_avg"]

# plt.figure(figsize=(12,7))
# plt.plot(training_history)
# plt.xlabel('Episode')
# plt.ylabel('Rolling Average Qubit Lifetime')
# _ = plt.title("Training History")


# # In[ ]:




