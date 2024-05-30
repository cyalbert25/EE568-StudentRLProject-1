import numpy as np
from abc import abstractmethod, ABC
from wur_environment import WUREnvironment

class EpsGreedyAgent(ABC):
    def __init__(self, env: WUREnvironment, explore_rate = 1.0):
        self._num_actions = env._num_actions
        self._num_states = env._num_states
        self._env = env

        # Initialize Q and N
        self._Q = np.zeros(shape=(self._num_states, self._num_actions), dtype=float)
        self._N = np.zeros((self._num_states, self._num_actions), dtype=int)

        self._step_counts = 0
        self._explore = explore_rate  
        self._reward_history = []

    def argmax_with_random_tie_breaking(self, b):
        return np.random.choice(np.where(b == b.max())[0])

    def select_action(self, state):
        # Should be selected on the basis of epsilon greedy from the current state
        explore = np.random.binomial(2, p=self._explore)
        if explore:
            # Exploration: With probability epsilon take a random action, an index of an action
            a = np.random.choice(np.arange(self._num_actions))
        else:
            # Exploitation: With probability 1 - epsilon take one of the optimal actions for the current state
            a = self.argmax_with_random_tie_breaking(self._Q[state, :])
        return a

    @abstractmethod
    def update_estimates(self, action, reward, prev_state):
        pass
    
    def get_current_average_reward(self):
        return np.mean(self._reward_history)
    
    def get_cumulative_reward(self):
        return np.sum(self._reward_history)
    

class QEpsGreedyAgent(EpsGreedyAgent):
    
    def update_estimates(self, action, reward, prev_state):
        """
        Updates the Q-value estimates based on the given action, reward, and previous state.

        Parameters:
            - action (int): The action taken in the previous state.
            - reward (float): The reward received after taking the action.
            - prev_state (int): The previous state.
        """
        curr_state = self._env._current_state

        # Interaction with the environment - collecting reward 
        self._reward_history.append(reward)

        self._N[prev_state, action] += 1
        alpha = 0.5 / (self._N[prev_state, action] + 1) ** 0.75

        # Update Q according to the algorithm
        self._Q[prev_state, action] = \
            (1 - alpha) * self._Q[prev_state, action] + alpha * (reward + self._env._gamma*np.max(self._Q[curr_state, :]))        

        self._step_counts += 1


class SARSAEpsGreedyAgent(EpsGreedyAgent):
    def update_estimates(self, action, reward, prev_state):
        """
        Updates the Q-value estimates based on the given action, reward, and previous state.

        Parameters:
            - action (int): The action taken in the previous state.
            - reward (float): The reward received after taking the action.
            - prev_state (int): The previous state.
        Returns:
            - new_action (int): The new action to take in the current state.
        """
        curr_state = self._env._current_state

        # Interaction with the environment - collecting reward 
        self._reward_history.append(reward)

        self._N[prev_state, action] += 1
        alpha = 0.5 / (self._N[prev_state, action] + 1) ** 0.75

        # Sampling a new action
        new_action = self.select_action(curr_state)
        # Update Q according to the algorithm
        self._Q[prev_state, action] = (1 - alpha) * self._Q[prev_state, action] + alpha * (reward + self._env._gamma*self._Q[curr_state, new_action])

        self._step_counts += 1
        return new_action