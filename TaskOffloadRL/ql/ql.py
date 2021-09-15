"""
Q Learning Algorithm
@Authors: TamNV
"""
import random
import numpy as np

class QLearning:

    """
    Implement QLearning Framework 
    """
    def __init__(self, state_dims, act_dims, alpha=0.0, gamma=0.0):
        """
        Initialize Method - for Q Learning
        param: state_dims: List 
        param: act_dims: List
        return None
        """
        self.state_dims = state_dims
        self.act_dims = act_dims
        self.alpha = alpha
        self.gamma = gamma

        self.state_act_dims = state_dims + act_dims
        self._build_model()

    def _build_model(self):
        """
        Define Model's Parameters
        """
        print("Shape of Model ", self.state_act_dims)
        self.model = np.zeros(np.prod(self.state_act_dims), np.float32) + -np.Inf
        self.model = np.reshape(self.model, self.state_act_dims)
        
    def _instane_update(self, state, next_state, action, reward, done):
        """
        Updated Model by One Observation 
        """
        state = [int(state[i]) for i in range(len(state))]
        next_state = [int(next_state[i]) for i in range(len(next_state))]
        action = [int(action[i]) for i in range(len(action))]
        action[0] = action[0] - 1
 
        idx_q = tuple(state + action) # Tuple
        state_q = self.model[idx_q]

        # Calculate Next State Value
        next_state_q = self.model[tuple(next_state)]
        next_state_q = np.max(next_state_q)

        if next_state_q == -np.Inf:
            next_state_q = 0.0

        if state_q == -np.Inf:
            state_q = 0.0
            alpha = 1.0
        else:
            alpha = self.alpha

        upd_value = (1 - alpha) * state_q + alpha * (reward + self.gamma * next_state_q * (1.0 - done))

        self.model[idx_q] = upd_value


    def _update_model(self, states, next_states, actions, rewards, dones):
        """
        Update Model's Parameters
        param: states: 2D Numpy Array - (batch_size x state_dims)
        param: next_states: 2D Numpy Array - (batch_size x state_dims)
        param: actions: 2D Numpy Array - (batch_size x act_dims)
        param: rewards: 1D Numpy Array - (batch_size)
        param: dones: 1D Numpy Array - (batch_size)
        """

        if states.shape[0] == next_states.shape[0]\
            == actions.shape[0] == rewards.shape[0]\
                == dones.shape[0]:
            pass
        else:
            print(states.shape[0], next_states.shape[0], actions.shape[0], rewards.shape[0], dones.shape[0])
            raise Exception("The Number Elements must be same!")
        
        num_samples = states.shape[0]

        for i in range(num_samples):
            state = np.copy(states[i])
            next_state = np.copy(next_states[i])
            action = np.copy(actions[i])
            reward = np.copy(rewards[i])
            done = np.copy(dones[i])

            self._instane_update(state, next_state, action, reward, done)
            
    def _sel_best_action(self, state):
        """
        Perform Selecting best Action for state
        """
        state = [int(state[i]) for i in range(len(state))]
        state = tuple(state)
        state_action_q = self.model[state]
        action = np.unravel_index(np.argmax(state_action_q, axis=None), self.act_dims)
        max_value = np.max(state_action_q)
        action = list(action)
        action[0] = action[0] + 1
        
        return action, max_value

#Action Selection Strategy
def sel_action(env, model, state, epsilon):
    """
    perform selecting action for Q Learning Method

    """
    if np.random.random() < epsilon:
        action = env.sam_action()
        return action
    else:
        action, value = model._sel_best_action(state)
        if value == -np.Inf:
            action = env.sam_action()
        return action
    