import numpy as np
from wur_environment import WUREnvironment
from utils import compute_subopt
from agents import QEpsGreedyAgent, SARSAEpsGreedyAgent, EpsGreedyAgent

NORMAL_STATE = 4

EC = 0
EA = 1
DO_NOTHING = 2

class User:

    @staticmethod
    def generate_users(base_transition_matrix, lambda_range, p_aha_range, p_aha_factor_range, posture_change_factor_range, num_users=10):
        users = []
        for _ in range(num_users):
            lambda_ = np.random.uniform(*lambda_range)
            p_aha = np.random.uniform(*p_aha_range, size=6)
            p_aha_factor = np.random.uniform(*p_aha_factor_range)
            posture_change_factor = np.random.uniform(*posture_change_factor_range)

            user = User(base_transition_matrix, lambda_, p_aha, p_aha_factor, posture_change_factor)
            users.append(user)
        return users

    def __init__(self, base_transition_model, lambda_, p_aha, p_aha_factor, posture_change_factor):
        self._lambda = lambda_
        self._p_aha = p_aha
        self._p_aha_factor = p_aha_factor
        self._posture_change_factor = posture_change_factor

        self._transition_model = self._initialize_transition_model(base_transition_model, posture_change_factor)
        self._env = WUREnvironment(self._transition_model, p_aha, lambda_, p_aha_factor)
        self._agent_q = None
        self._agent_sarsa = None


    def _initialize_transition_model(self, base_transition_model, posture_change_factor):
        num_states = base_transition_model.shape[1]
        transition_model = base_transition_model.copy()

        # Modify the EA transition
        transition_model[EA] += np.random.normal(0, 0.05, size=(num_states, num_states))
        transition_model[EA] = np.abs(transition_model[EA])
        transition_model[EA] /= np.sum(transition_model[EA], axis=1)[:, np.newaxis]
        assert np.sum(transition_model[EA], axis=1).all() == 1

        transition_model[EA, NORMAL_STATE] = np.zeros(num_states)
        transition_model[EA, NORMAL_STATE, NORMAL_STATE] = 1
                                                                     
        # Modify the DO_NOTHING transition
        for i in range(num_states):
            transition_model[DO_NOTHING][i, i] *= posture_change_factor
            
            mask = np.ones(num_states, dtype=bool)
            mask[i] = False
            normalization_factor = (1 - transition_model[DO_NOTHING][i, i]) / np.sum(transition_model[DO_NOTHING][i, mask])
            transition_model[DO_NOTHING][i, mask] *= normalization_factor
        assert np.sum(transition_model[DO_NOTHING], axis=1).all() == 1
        
        return transition_model
    
    def run_simulation(self, agent: EpsGreedyAgent, num_iterations: int = 500, t_timeout: int = 7):
        self._env.reset_environment()

        subopts = []
        average_rewards = []
        cum_rewards = []
        idxs = []

        counter = 0
        action = agent.select_action(self._env._current_state) if isinstance(agent, SARSAEpsGreedyAgent) else None
        while counter < num_iterations:
            state = self._env._current_state

            # If the current state is normal, let agent do nothing and continue to the next iteration
            if state == NORMAL_STATE:
                self._env.step(action=DO_NOTHING)
                counter += 1
                continue

            # Else, select the corresponding action
            action = action if isinstance(agent, SARSAEpsGreedyAgent) else agent.select_action(state)
            reward = -1
            for idx in range(t_timeout):
                self._env.step(action)
                if self._env._current_state == NORMAL_STATE:
                    reward = 1
                    break

            if isinstance(agent, SARSAEpsGreedyAgent):
                action = agent.update_estimates(action, reward, state)
            else:
                agent.update_estimates(action, reward, state)
            
            average_rewards.append(agent.get_current_average_reward())
            cum_rewards.append(agent.get_cumulative_reward())
            idxs.append(idx + 1)
            subopts.append(compute_subopt(self._env, agent))

            counter += idx + 1

        return subopts, average_rewards, cum_rewards, idxs

            
    def simulation(self, algorithm:str, num_iterations=500, explore_rate=1.0, t_timeout=20):
        if algorithm == "Q":
            agent = self._agent_q = QEpsGreedyAgent(self._env, explore_rate)
        elif algorithm == "SARSA":
            agent = self._agent_sarsa = SARSAEpsGreedyAgent(self._env, explore_rate)
        else:
            ValueError(f"Invalid algorithm '{algorithm}' chosen.")

        return self.run_simulation(agent, num_iterations, t_timeout)                 
    
    def optimal_actions(self, Q):
        actions = {0: "EC", 1: "EA", 2: "NOTHING"}
        optimal_action_idx = np.argmax(Q, axis=1)
        optimal_actions = np.array([actions[i] for i in optimal_action_idx])
        
        return np.array([optimal_actions[:3], optimal_actions[3:]])