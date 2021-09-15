"""
Task Offload Environment
@Authors: TamNV
"""
import copy
import random
import numpy as np

class Helper:

    def __init__(self, rc, max_f, max_c):
        """
        Initial Method for Continuous Environment
        params - rc: float - Radius Cycle
        params - max_f: float - Maximum CPUs
        params - max_c: float - Maximum Costs
        """
        if rc < 0 or max_f < 0 or max_c < 0:
            raise Exception(
                    "Initial Values for Helper must be Positive!"
                )
        self.rc = rc
        self.max_f = max_f
        self.max_c = max_c

        self.d = None
        self.f = None
        self.c = None

    def become_stranger(self):
        """This node become to a Stranger Node"""
        f_frac = np.random.uniform(low=1e-6, high=1e-5)
        c_frac = np.random.uniform(low=0.8, high=1.0)
        self.f = self.f * f_frac

        if np.random.rand() < 0.9:
            self.c = self.c  * f_frac * np.random.uniform(low=10.0, high=20.0)
        else:
            self.c = self.c * f_frac

    def reset(self):
        """
        Create a New Instance
        """
        self.d = np.random.uniform(low=1e-3 * self.rc, high=self.rc)
        self.f = np.random.normal(loc=self.max_f * 0.5, scale=self.max_f * 1e-2)
        self.c = np.random.normal(loc=self.max_c * 0.5, scale=self.max_c * 1e-2)

        if np.random.rand() < 0.1:
            self.become_stranger()

        if self.f < 0:
            self.f = self.max_f * 0.5
        if self.c < 0:
            self.c = self.max_c * 0.5

    def transit(self):
        """
        Move to a Next State
        """
        self.d = np.random.normal(loc=self.d, scale=self.rc * 1e-3)
        self.f = np.random.normal(loc=self.f, scale=self.max_f * 1e-2)
        self.c = np.random.normal(loc=self.c, scale=self.max_c * 1e-2)

        if self.d < 0:
            self.d = self.rc * 0.5

        if self.f < 0:
            self.f = self.max_f * 0.5

        if self.c < 0:
            self.c = self.max_c * 0.5

        if np.random.rand() < 0.1:
            self.become_stranger()

    def cal_com_latency(self, num_bytes):
        """
        Calculate The Latency for Computing "num_bytes" data 
        params: num_bytes - Integer - Computation Demand
        """
        num_bytes = float(num_bytes)
        latency = num_bytes / self.f
        return latency

    def cal_offload_latency(self, num_bytes):
        """
        Calculate The Latency for Offloading "num_bytes" data
        params: num_bytes - Integer - Offloading Demand
        """
        num_bytes = float(num_bytes)
        # Transformation Parameters
        CO = 3*1e8
        FB = 1e8
        B = 40.0*1e12
        PT = 0.25
        sigma = 0.5

        pr = PT * (CO** 2) / (((np.pi ** 2) * FB * self.d) ** 2)
        rn = B * np.log(1.0 + pr /sigma) 
        latency = num_bytes / rn
        return latency

    def cal_incentive_cost(self, num_bytes):
        """
        Calculate the Incentive Cost for Processing "num_bytes" data
        params: - num_bytes : Integer
        """
        cost = self.c * self.cal_com_latency(num_bytes)
        return cost

    def show_cur_state(self):
        print("d: {:.3f}, f: {:.3f}, c: {:.10f}".format(self.d, self.f, self.c))

    def get_state(self):
        """
        Get the Current State of This Helper
        """
        state = [self.f/self.max_f, self.c/self.max_c, self.d/self.rc]

        return state

class TaskOffloadEnv:

    def __init__(self, n_helpers, rc, max_f, max_c, max_l, alpha1, alpha2, seed=1):
        """
        Initial Method for Task Offload Environments
        """
        self.n_helpers = n_helpers
        self.rc = rc
        self.max_f = max_f
        self.max_c = max_c
        self.max_l = max_l
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.helpers = {}
        self.step_counter = 0
        # environment's configuration
        self.act_dims = [self.n_helpers] + [2 for _ in range(self.n_helpers)]
        self.num_actions = self.n_helpers * (2 ** self.n_helpers)
        self.env_dims = 1 + self.n_helpers * 3

        np.random.seed(seed)

    def get_state(self):
        """
        Get Environment State
        """
        client_state = [self.l*1.0/self.max_l]
        helper_state = []

        for key in sorted(list(self.helpers.keys())):
            helper = self.helpers[key]
            state = helper.get_state()
            helper_state += state
        env_state = client_state + helper_state
        return env_state

    def reset(self):
        """
        Create a New Instance
        """
        self.l = np.random.normal(loc=self.max_l * 0.5, scale=self.max_l * 1e-4)
        if self.l < 0:
            self.l = self.max_l * 0.5
        self.step_counter = 0
        for idx in range(self.n_helpers):
            self.helpers[idx] = Helper(self.rc, self.max_f, self.max_c)
            self.helpers[idx].reset()
            
        self.client_f = self.max_f * np.random.normal(loc=0.1, scale= 1e-4)

        state = self.get_state()
        return state

    def step(self, action):
        """
        Perform an action
        action's format [k, a1, ..., aN]

        """
        k = action[0]
        a_vec = action[-self.n_helpers:]
        m = sum(a_vec)

        standard_time = self.l / self.client_f
        self.step_counter += 1
        
        com_fee, total_latency = [], []
        num_bytes = self.l / k
        for idx in sorted(list(self.helpers.keys())):
            if a_vec[idx] == 0:
                total_latency.append(np.Inf)
                com_fee.append(0.0)
            else:
                helper = self.helpers[idx]
                offload_latency = helper.cal_offload_latency(num_bytes)*0.0
                com_latency = helper.cal_com_latency(num_bytes)
                fee = helper.cal_incentive_cost(num_bytes)
                com_fee.append(fee)
                total_latency.append(offload_latency + com_latency)

        total_latency = sorted(total_latency)
        required_latency = max(total_latency[:k])
        required_fee = np.sum(com_fee)
        done = False
        # Calculate in Case the action meets the conditions
        if k <= m:
            if required_latency > standard_time:
                com_reward = -standard_time
            else:
                com_reward = standard_time - required_latency
            com_reward = com_reward * self.alpha1
            cost_reward = required_fee * self.alpha2
            total_reward = com_reward - cost_reward
        else:
            """
            an action doesn't meet the conditions
            """
            com_reward = -1.0 * self.alpha1 * standard_time
            cost_reward = 1.0 * self.alpha2 * required_fee
            if m == 0:
                cost_reward = (self.l / self.client_f) * self.max_c
            total_reward = com_reward - cost_reward
        reward = [total_reward, com_reward, cost_reward]
        """
        Move to the next State
        """
        # Check Finishing Round
        if self.step_counter == 50:
            done = True
        self.l = np.random.normal(loc=self.l, scale=self.max_l * 1e-4)
        if self.l < 0:
            self.l = self.max_l * 0.5
        for key in self.helpers.keys():
            self.helpers[key].transit()
        next_state = self.get_state()
        return next_state, reward, done

    def sam_action(self):
        """
        select one action randomly
        action's format [k, a1, ..., aN]
        """
        k = random.randint(1, self.n_helpers)
        # => 3
        n = random.randint(1, self.n_helpers)
        # => 4
        a_vec = [0.0 for _ in range(self.n_helpers)]
        # =>[0 0 0 0 0]
        sel_helper_idxs = np.random.permutation(self.n_helpers)[0:n]
        # [0, 4 , 1, 3, 2] => [0, 4 , 1, 3]
        for helper_idx in list(sel_helper_idxs):
            a_vec[helper_idx] = 1.0
        action = [k] + a_vec
        return action

    def action2index(self, action):
        """
        Convert action from nulti-dimension format to index format
        """
        if len(action) != len(self.act_dims):
            raise Exception("Shape Error")
        
        act_idx = action[0] - 1
        
        for i in range(1, len(self.act_dims)):
            act_idx = act_idx * self.act_dims[i] + action[i]

        return act_idx

    def index2action(self, act_idx):
        """
        Convert action from index format to multi-dimension format
        """
        action = []
        for i in range(len(self.act_dims), 1, -1):
            ai = act_idx % self.act_dims[i-1]
            action.append(int(ai))
            act_idx = (act_idx - ai) / self.act_dims[i-1]
        
        action.append(int(act_idx))
        action.reverse()
        action[0] = action[0] + 1
        return action

if __name__ == "__main__":
    env = TaskOffloadEnv(n_helpers=5,
                        rc=100,
                        max_f=100,
                        max_c=100,
                        max_l=100,
                        alpha1=1.0,
                        alpha2=1.0,
                        seed=1)

    action = env.sam_action()
    idx = env.action2index(action)
    recovered = env.index2action(idx)

    print(action, recovered)

