"""
Deep Q Learning Method is Implemented Here
@Authors: TamNV
"""
import numpy as np
import tensorflow as tf

# Backbone Architecture
class Backbone:
	
	def __init__(self, out_dims):
		
		self.layers = [
			tf.keras.layers.Dense(units=64, activation=tf.nn.relu),
			tf.keras.layers.Dense(units=64, activation=tf.nn.relu),
			tf.keras.layers.Dense(units=64, activation=tf.nn.relu),
			tf.keras.layers.Dense(units=out_dims)
		]
	
	def __call__(self, inputs):
		# Call Function
		x = inputs
		outputs = self.call(inputs)
		return outputs
	
	def call(self, inputs):
		# Perform
		x = inputs
		for layer in self.layers:
			x = layer(x)
		return x

class DQL:
	def __init__(self, input_dims, num_actions, backbone):
		"""
		Initial Method for Deep Q Learning Method
		params: input_dims: Integer
		params: num_actions: Integer
		returns: None
		"""
		self.input_dims = input_dims
		self.num_actions = num_actions
		self.backbone = backbone
		self.states = tf.compat.v1.placeholder(dtype=tf.float32,
							shape=(None, self.input_dims))
		self.next_states = tf.compat.v1.placeholder(dtype=tf.float32,
								shape=(None, self.input_dims))
		self.actions = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None,))
		self.rewards = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None,))
		self.dones = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, ))
		self.gamma = tf.compat.v1.placeholder(dtype=tf.float32, shape=None)
		self.lr = tf.compat.v1.placeholder(dtype=tf.float32, shape=None)

		self._define_model()

	def _define_model(self):
		# Create the Deep Q Learning agent
		self.agent = self.backbone(out_dims=self.num_actions)
		# convert action to one-hot format
		one_hot_actions = tf.one_hot(self.actions, self.num_actions)
		self.q_values = self.agent(self.states)
		q_action_values = tf.reduce_sum(one_hot_actions * self.q_values , axis=-1)
		# Using Bellman Equation to Caculate Ground-Truth
		q_next_values = self.agent(self.next_states)
		q_next_values = tf.reduce_max(q_next_values, axis=-1)
		label = (q_next_values * (1.0 - self.dones)) * self.gamma + self.rewards
		# define Objective Function
		self.loss = tf.losses.mean_squared_error(label, q_action_values)
		self.opt = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(loss=self.loss)

def sel_action(env, model, sess, state, epsilon):
	"""
	perform selecting action for Deep Q Learning Method

	"""
	if np.random.random() < epsilon:
		action = env.sam_action()
		return action
	else:
		state = np.expand_dims(state, axis=0)  
		# Run Inference
		[q_values] = sess.run([model.q_values],
						feed_dict={model.states: state})
		q_values = q_values[0]
		act_idx = np.argmax(q_values)
		action = env.index2action(act_idx)
		return action

if __name__ == "__main__":
	model = DQL(input_dims=8, num_actions=10, backbone=Backbone)
