"""
Monte Carlo Method
@Authors: TamNV 
"""
import os
import json
import time
import argparse
import numpy as np

from env import TaskOffloadEnv

parser = argparse.ArgumentParser(description='')
parser.add_argument('--config_file', dest='config_file',\
	help=" ", default="../config/helper_03.json")
args = parser.parse_args()

def main(config):
	env = TaskOffloadEnv(n_helpers=config["n_helpers"],
				rc=config["rc"],
				max_f=config["max_f"],
				max_c=config["max_c"],	
				max_l=config["max_l"],
				alpha1=config["alpha1"],
				alpha2=config["alpha2"],
				seed=1)

	# Run Random Wark Algorithm
	log_total_reward, log_comp_reward, log_cost_reward = [], [], []

	for episode in range(config["num_episodes"]):
		# Perfome Evaluation
		# exp_times = []
		if (episode) % 50 == 0:
			
			avg_total, avg_comp, avg_cost = [], [], []
			
			for game in range(10):
				state = env.reset()
				done = False
				total_reward, comp_reward, cost_reward = 0, 0, 0
				
				while not done:
					# start_time = time.time()
					action = env.sam_action()
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
			
			print("Episode {} - Total Reward {:.5f} - Computation Reward {:.5f} Cost Reward {}".\
				format(episode, avg_total, avg_comp, avg_cost))
		# print("N Helpers: {}; Consuming time per action : {:8f}".format(config["n_helpers"], np.mean(exp_times)))
	rw_log = {
		"total": log_total_reward,
		"computation": log_comp_reward,
		"cost": log_cost_reward,
	}

	name = "normal_helper_{}_lmax_{}_u{}_v{}.json".format(config["n_helpers"], int(config["max_l"]),\
		int(config["alpha1"]), int(config["alpha2"]))
	log_file = os.path.join(config["log_dir"], name)

	with open(log_file, "w") as f:
		json.dump(rw_log, f, indent=4)

if __name__ == "__main__":
	# Read Configuration File
	print("Configuration File {}".format(args.config_file))
	with open(args.config_file, "r") as file:
		config = json.load(file)

	assert config is not None, "Meeting Issues???"

	main(config)

	
