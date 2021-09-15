"""
Results Evaluation is Implemented here
@Authors: TamNV
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np

def vil_log_results(values, names, colors, x_name, y_name, save_file):
	"""
	Result Visualization
	"""

	plt.clf()
	fig, ax = plt.subplots(1,1)
	fig.set_size_inches(4, 2.5)

	idxs = list(range(len(values[0])))
	for i in range(len(values)):
		ax.plot(idxs, values[i], color=colors[i], label=names[i], linewidth=1.75)
	
	ax.set_xlabel(x_name)
	ax.set_ylabel(y_name)
	
	_idxs = np.linspace(0, len(idxs)-1, 7)
	x_idxs = [idxs[int(i)] for i in _idxs]
	x_names = np.array(x_idxs) * 50
	
	plt.xticks(ticks=x_idxs, labels=x_names)
	
	plt.tight_layout()
	plt.legend(loc=4)
	plt.grid()
	plt.savefig(save_file, bbox_inches='tight')

def compare_hyper_parameters(rewards,
						x_axis_name="L Size",
						y_axis_name="Total Reward",
						x_names= ["L", "10L", "50L", "100L"],
						color=["teal", "navy", "forestgreen"],
						labels=["DRL", "QL", "RM"],
						linestyles = ["-", ":", "-."],
						maskers=['o', 'v', "^"],
						save_file="images/l_comparison.png"):
	
	
	plt.clf()
	fig, ax = plt.subplots(1,1)
	fig.set_size_inches(4, 2.5)
	for idx, reward in enumerate(rewards):
		idxs = np.arange(len(reward))
		ax.plot(idxs,\
				reward,\
				color=color[idx],\
				marker=maskers[idx],\
				linestyle=linestyles[idx],\
				label=labels[idx])
	
	ax.set_xlabel(x_axis_name)
	ax.set_ylabel(y_axis_name)

	x_idxs = np.arange(len(x_names))

	plt.xticks(ticks=x_idxs, labels=x_names)
	
	plt.tight_layout()
	plt.legend(loc=0)
	plt.grid()
	plt.savefig(save_file, bbox_inches='tight')

def load_data(json_file):
	"""
	Load data from json file
	"""
	with open(json_file, "r") as file:
		content = json.load(file)

	assert content is not None, "Meeting Issues???"
	return content

if __name__ == "__main__":
	log_dir, n_eps = "log", 4

	colors = ["red", "blue", "green"]
	# compare total reward - dql, ql, rw
	file_names = [
					"dql_helper_3_lmax_3000000_u1_v1.json",
					"ql_helper_3_lmax_3000000_u1_v1.json",
					"normal_helper_3_lmax_3000000_u1_v1.json",
				]

	total_rewards = []
	for file_name in file_names:
		file_name = os.path.join(log_dir, file_name)
		data = load_data(file_name)
		total = data["total"]
		total_rewards.append(total)

	vil_log_results(values=total_rewards,
					names=["DRL", "QL", "RM"],
					colors=colors,
					x_name="Episode",
					y_name="Total Reward",
					save_file="image/total.png")

	# Compare Diffirent Size of Computation Demand
	file_names = [
				"dql_helper_3_lmax_3000000_u1_v1.json",
				"dql_helper_3_lmax_30000000_u1_v1.json",
				"dql_helper_3_lmax_150000000_u1_v1.json",
				"dql_helper_3_lmax_300000000_u1_v1.json"
			]
	
	dql_total_rewards = []
	for file_name in file_names:
		file_name = os.path.join(log_dir, file_name)
		data = load_data(file_name)
		dql_total_rewards.append(np.mean(sorted(data["total"])[-n_eps:]))

	file_names = [
				"ql_helper_3_lmax_3000000_u1_v1.json",
				"ql_helper_3_lmax_30000000_u1_v1.json",
				"ql_helper_3_lmax_150000000_u1_v1.json",
				"ql_helper_3_lmax_300000000_u1_v1.json"
			]
			
	ql_total_rewards = []
	for file_name in file_names:
		file_name = os.path.join(log_dir, file_name)
		data = load_data(file_name)
		ql_total_rewards.append(np.mean(sorted(data["total"])[-n_eps:]))

	file_names = [
			"normal_helper_3_lmax_3000000_u1_v1.json",
			"normal_helper_3_lmax_30000000_u1_v1.json",
			"normal_helper_3_lmax_150000000_u1_v1.json",
			"normal_helper_3_lmax_300000000_u1_v1.json"
		]

	rw_total_rewards = []
	for file_name in file_names:
		file_name = os.path.join(log_dir, file_name)
		data = load_data(file_name)
		rw_total_rewards.append(np.mean(sorted(data["total"])[-n_eps:]))

	compare_hyper_parameters([dql_total_rewards, ql_total_rewards, rw_total_rewards],
						x_axis_name="Task Size (L)",
						y_axis_name="Total Reward",
						color=colors,
						save_file="image/l_size.png")

	# # Compare Diffirent Helpers
	drl_file_names = [
				"dql_helper_3_lmax_3000000_u1_v1.json", # 0
				"dql_helper_4_lmax_3000000_u1_v1.json", # 1
				"dql_helper_5_lmax_3000000_u1_v1.json", # 2
				"dql_helper_6_lmax_3000000_u1_v1.json", # 3
				"dql_helper_7_lmax_3000000_u1_v1.json", # 4
				"dql_helper_8_lmax_3000000_u1_v1.json", # 5
				"dql_helper_9_lmax_3000000_u1_v1.json", # 6
				"dql_helper_10_lmax_3000000_u1_v1.json", #7
			]
	converge_indexs = [
		[57, 59],
		[55, 58],
		[58, 60],
		[45, 48],
		[33, 35],
		[55, 57],
		[23, 25],
		[58, 60]
	]

	dql_computations, dql_costs = [], []
	
	for idx, file_name in enumerate(drl_file_names):
		file_name = os.path.join(log_dir, file_name)
		data = load_data(file_name)
		dql_computations.append(data["computation"])
		dql_costs.append(data["cost"])

	for idx in range(len(drl_file_names)):
		start, stop = converge_indexs[idx]
		print(drl_file_names[idx])
		print(dql_computations[idx][start:stop])
		print(dql_costs[idx][start:stop])
		dql_computations[idx] = np.mean(dql_computations[idx][start:stop])
		dql_costs[idx] = np.mean(dql_costs[idx][start:stop])


		
	
	ql_file_names = [
				"ql_helper_3_lmax_3000000_u1_v1.json",
				"ql_helper_4_lmax_3000000_u1_v1.json",
				"ql_helper_5_lmax_3000000_u1_v1.json",
				"ql_helper_6_lmax_3000000_u1_v1.json",
				"ql_helper_7_lmax_3000000_u1_v1.json",
				# "ql_helper_8_lmax_3000000_u1_v1.json",
				# "ql_helper_9_lmax_3000000_u1_v1.json",
				# "ql_helper_10_lmax_3000000_u1_v1.json",
			]

	ql_computations, ql_costs = [], []
	
	for idx, file_name in enumerate(ql_file_names):
		file_name = os.path.join(log_dir, file_name)
		data = load_data(file_name)
		ql_computations.append(np.mean(data["computation"][-n_eps:]))
		ql_costs.append(np.mean(data["cost"][-n_eps:]))

	rw_file_names = [
				"normal_helper_3_lmax_3000000_u1_v1.json",
				"normal_helper_4_lmax_3000000_u1_v1.json",
				"normal_helper_5_lmax_3000000_u1_v1.json",
				"normal_helper_6_lmax_3000000_u1_v1.json",
				"normal_helper_7_lmax_3000000_u1_v1.json",
				"normal_helper_8_lmax_3000000_u1_v1.json",
				"normal_helper_9_lmax_3000000_u1_v1.json",
				"normal_helper_10_lmax_3000000_u1_v1.json",
			]

	rw_computations, rw_costs = [], []

	for idx, file_name in enumerate(rw_file_names):
		file_name = os.path.join(log_dir, file_name)
		data = load_data(file_name)

		rw_computations.append(np.mean(data["computation"][-n_eps:]))
		rw_costs.append(np.mean(data["cost"][-n_eps:]))

	compare_hyper_parameters([dql_computations, ql_computations, rw_computations],
						x_axis_name="Number of Helpers (N)",
						y_axis_name="Computation Reward",
						x_names= ["3", "4", "5", "6", "7", "8", "9", "10"],
						color=colors,
						save_file="image/helper_com.png")
	compare_hyper_parameters([dql_costs, ql_costs, rw_costs],
						x_axis_name="Number of Helpers (N)",
						y_axis_name="Incentive Reward",
						x_names= ["3", "4", "5", "6", "7", "8", "9", "10"],
						color=colors,
						save_file="image/helper_cost.png")
	
	dql_times=np.array([0.000305, 0.000305, 0.000341, 0.000318, 0.000346, 0.000354, 0.00042, 0.000498])
	ql_times=np.array([0.000049, 0.000052, 0.000053, 0.000056, 0.000056])	
	rw_times=np.array([0.000017, 0.000019, 0.00002, 0.000016, 0.000017, 0.000017, 0.000017, 0.000021])
	compare_hyper_parameters([dql_times, ql_times, rw_times],
						x_axis_name="Number of Helpers (N)",
						y_axis_name="Time Consuming (s)",
						x_names= ["3", "4", "5", "6", "7", "8", "9", "10"],
						color=colors,
						save_file="image/time_consuming.png")
# # Compare Diffirent Helpers
	drl_file_names = [
		"dql_helper_3_lmax_3000000_u1_v1.json",
		"dql_helper_3_lmax_3000000_u1_v10.json",
		"dql_helper_3_lmax_3000000_u1_v20.json",
		"dql_helper_3_lmax_3000000_u1_v30.json",
		"dql_helper_3_lmax_3000000_u1_v40.json"
	]

	dql_computations, dql_costs = [], []
	
	for idx, file_name in enumerate(drl_file_names):
		file_name = os.path.join(log_dir, file_name)
		data = load_data(file_name)
		dql_computations.append(data["computation"])
		dql_costs.append(data["cost"])

	for idx in range(len(drl_file_names)):
		if idx == 0:
			start, stop = 57, 59
		elif idx == 1:
			start, stop = 6, 9
		elif idx == 2:
			start, stop = 32, 33
		elif idx == 3:
			start, stop = 25, 29
		elif idx == 4:
			start, stop = 57, 59
		dql_computations[idx] = np.mean(dql_computations[idx][start:stop])
		dql_costs[idx] = np.mean(dql_costs[idx][start:stop])
		
	
	ql_file_names = [
					"ql_helper_3_lmax_3000000_u1_v1.json",
					"ql_helper_3_lmax_3000000_u1_v10.json",
					"ql_helper_3_lmax_3000000_u1_v20.json",
					"ql_helper_3_lmax_3000000_u1_v30.json",
					"ql_helper_3_lmax_3000000_u1_v40.json"
			]

	ql_computations, ql_costs = [], []
	
	for idx, file_name in enumerate(ql_file_names):
		file_name = os.path.join(log_dir, file_name)
		data = load_data(file_name)
		if idx == 0:
			start, stop = 55, 59
		if idx == 1:
			start, stop = 55, 59
		if idx == 2:
			start, stop = 55, 59
		if idx == 3:
			start, stop = 55, 59
		if idx == 4:
			start, stop = 55, 59
		ql_computations.append(np.mean(data["computation"][start:stop]))
		ql_costs.append(np.mean(data["cost"][start:stop]))

	rw_file_names = [
					"normal_helper_3_lmax_3000000_u1_v1.json",
					"normal_helper_3_lmax_3000000_u1_v10.json",
					"normal_helper_3_lmax_3000000_u1_v20.json",
					"normal_helper_3_lmax_3000000_u1_v30.json",
					"normal_helper_3_lmax_3000000_u1_v40.json"
			]

	rw_computations, rw_costs = [], []

	for idx, file_name in enumerate(rw_file_names):
		file_name = os.path.join(log_dir, file_name)
		data = load_data(file_name)

		rw_computations.append(np.mean(data["computation"][-n_eps:]))
		rw_costs.append(np.mean(data["cost"][-n_eps:]))

	compare_hyper_parameters([dql_computations, ql_computations, rw_computations],
						x_axis_name="The ratio v/u",
						y_axis_name="Computation Reward",
						x_names= ["1", "10", "20", "30", "40"],
						color=colors,
						save_file="image/helper_com_over_u_v.png")

	compare_hyper_parameters([dql_costs, ql_costs, rw_costs],
						x_axis_name="The ratio v/u",
						y_axis_name="Computation Reward",
						x_names= ["1", "10", "20", "30", "40"],
						color=colors,
						save_file="image/helper_cost_over_u_v.png")
