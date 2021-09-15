"""
Main Function Deep Q Learning
@Authors: TamNV
"""
import os
import time
import json
import argparse
import numpy as np

from env import TaskOffloadEnv
from dql import *
from relay import Memory

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

parser = argparse.ArgumentParser(description='')
parser.add_argument('--config_file', dest='config_file',\
	help=" ", default="../config/helper_03_01.json")
args = parser.parse_args()

def main(config):
	config["gamma"] = 0.99
	env = TaskOffloadEnv(n_helpers=config["n_helpers"],
				rc=config["rc"],
				max_f=config["max_f"],
				max_c=config["max_c"],
				max_l=config["max_l"],
				alpha1=config["alpha1"],
				alpha2=config["alpha2"],
				seed=1)

	model = DQL(input_dims=env.env_dims,
				num_actions=env.num_actions,
				backbone=Backbone)

	os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu_idx"])
	gpu_option = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_option))
	sess.run(tf.global_variables_initializer())

	container = Memory(config["buffer_size"])
	log_total_reward, log_comp_reward, log_cost_reward = [], [], []
	epsilon, lr = config["max_eps"], config["lr"]

	for episode in range(config["num_episodes"]):
		# Perform Evaluation
		# exp_times = []
		if (episode) % 50 == 0:
			avg_total, avg_comp, avg_cost = [], [], []
			for game in range(10):
				state = env.reset()
				done = False
				total_reward, comp_reward, cost_reward = 0, 0, 0
					
				while not done:
					# start_time = time.time()
					action = sel_action(env, model, sess, state, -np.Inf)
					# end_time = time.time()
					# exp_times.append(end_time - start_time)

					next_state, reward, done = env.step(action)
					state = next_state
						
					total_reward += reward[0]
					comp_reward += reward[1] * 1.0 / config["alpha1"]
					cost_reward += reward[2] * 1.0 / config["alpha2"]
					
				avg_total.append(total_reward)
				avg_comp.append(comp_reward)
				avg_cost.append(cost_reward)
					
			avg_total = np.mean(avg_total)
			avg_comp = np.mean(avg_comp)
			avg_cost = np.mean(avg_cost)
				
			log_total_reward.append(avg_total)
			log_comp_reward.append(avg_comp)
			log_cost_reward.append(avg_cost)
			# if (episode) % 100 == 0:
			print("Episode {} - Total Reward {:.5f} - Computation Reward {:.5f} Cost Reward {}".\
				format(episode, avg_total, avg_comp, avg_cost))
		# print("N Helpers: {}; Consuming time per action : {:7f}".format(config["n_helpers"], np.mean(exp_times)))
		# break
		# Perform Evaluation
		state = env.reset()
		done = False
		if (episode + 1) % 50 == 0:
			epsilon *= 0.99
		
		epsilon = max(epsilon, config["min_eps"])
		
		while not done:
			action = sel_action(env, model, sess, state, epsilon)
			next_state, reward, done = env.step(action)
			container.insert_samples({'s': [state],
									'a':[env.action2index(action)],
									'ns':[next_state],
									'r':[float(reward[0])],
									'd':[float(0.0)]})
			state = next_state
		
		batch_data = container.sel_samples(config["batch_size"])
		states = np.array(batch_data["s"])
		actions = np.array(batch_data["a"])
		next_states = np.array(batch_data["ns"])
		rewards = np.array(batch_data["r"])
		dones = np.array(batch_data["d"])

		[_, loss] = sess.run([model.opt, model.loss],
							feed_dict={
								model.states: states,
								model.next_states:next_states,
								model.actions: actions,
								model.rewards: rewards,
								model.dones: dones,
								model.gamma: config["gamma"],
								model.lr: lr
							})
		
	ql_log = {
		"total": log_total_reward,
		"computation": log_comp_reward,
		"cost": log_cost_reward
	}

	# Save Log Information
	name = "dql_helper_{}_lmax_{}_u{}_v{}.json".format(config["n_helpers"], int(config["max_l"]), int(config["alpha1"]), int(config["alpha2"]))
	log_file = os.path.join(config["log_dir"], name)

	with open(log_file, "w") as f:
		json.dump(ql_log, f, indent=4)
		
if __name__ == "__main__":
	# Read Configuration File
	print("Configuration File {}".format(args.config_file))
	with open(args.config_file, "r") as file:
		config = json.load(file)
	assert config is not None, "Meeting Issues???"
	main(config)