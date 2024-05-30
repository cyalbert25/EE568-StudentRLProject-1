import numpy as np

NORMAL_STATE = 4

EC = 0
EA = 1
DO_NOTHING = 2

class WUREnvironment:

    def __init__(self, transition_model, p_aha, lambda_=0.01, p_aha_factor = 1.001):
        # Defining the number of states
        self._num_states = 6
        # Defining the number of actions
        self._num_actions = 2
        # Discount factor
        self._gamma = 0.99

        # Decaying factor for DO NOTHING action in transition matrix
        self._lambda = lambda_
        # By how much to increase the probability of AHA moment when the user gets it
        self._p_aha_factor = p_aha_factor

        # User specifications - transition matrix
        self._T = transition_model.copy() # shape = (num_actions, num_states, num_states); Dependent on the user -> given externally
        # Saves the copy as a starting point for subsequent p_aha modifications in the _step method
        self._orig_T = transition_model.copy() 
        
        # User specifications - p_aha
        self._p_aha = p_aha.copy()
        # Saves the copy as a starting point for subsequent p_aha modifications in the _step method
        self._orig_p_aha = p_aha.copy()
        
        self._current_state = NORMAL_STATE # the default (normal, normal) state
        self._step_count = 0

        self._rewards = np.array([
            [-1, -1],
            [-1, -1],
            [-1, -1],
            [-1, -1],
            [1, 1],
            [-1, -1]
        ])

    def reset_environment(self):
        self._T = self._orig_T.copy()
        self._current_state = NORMAL_STATE
        self._step_count = 0
        self._p_aha = self._orig_p_aha.copy()

    def step(self, action):
        """
        Takes an action and updates the current state of the environment.

        Parameters:
        - action (int): The action to be taken.
        Returns:
        - previous_state (int): The state before the update.
        """
        self._update_ea_paha()
        new_state = np.random.choice(np.arange(self._num_states), p=self._T[action][self._current_state])

        if action == EA:
            self._p_aha[self._current_state] *= self._p_aha_factor
            assert self._p_aha[self._current_state] < 0.6

        previous_state = self._current_state
        self._current_state = new_state
        
        self._decay_do_nothing()
        self._step_count += 1
        
        return previous_state

    def _fatigue(self):
        """
        We model the fatigue factor as e^{-lambda}. It multiplies the probabilities of staying in the same state when no action is taken.
        """
        return np.exp(-self._lambda)

    def _decay_do_nothing(self):
        for i in range(self._num_states):
            decay_update = self._T[DO_NOTHING][i, i] * self._fatigue()
            # assert decay_update > 0.5
            decay_update = min(decay_update, 0.5)
            self._T[DO_NOTHING][i, i] = decay_update

            mask = np.ones(self._num_states, dtype=bool)
            mask[i] = False
            normalization_factor = (1 - self._T[DO_NOTHING][i, i]) / np.sum(self._T[DO_NOTHING][i, mask])
            self._T[DO_NOTHING][i, mask] *= normalization_factor
            
        assert np.sum(self._T[DO_NOTHING], axis=1).all() == 1

    def _update_ea_paha(self):
        self._T[EA] = self._orig_T[EA] * (1 - self._p_aha).reshape(-1, 1)
        self._T[EA, :, NORMAL_STATE] += self._p_aha
        assert self._T[EA].sum(axis=1).all() == 1