import numpy as np

NORMAL_STATE = 4
EA = 1

def value_iteration(env, tol: float = 1e-10):
    """
    Perform value iteration to find the optimal value function for the given environment.

    Parameters:
    - env: WUREnvironment
        The environment for which the value function is computed.
    - tol: float, optional (default=1e-10)
        The tolerance for convergence criterion.

    Returns:
    - v: numpy.ndarray
        The optimal value function for the given environment.
    """
    v = np.zeros(env._num_states) # initialize value function
    q = np.zeros((env._num_states, env._num_actions)) #initialize Q-value

    T = env._T.copy()
    T = np.linalg.matrix_power(T, 7)

    # while True:
    #     v_old = np.copy(v) # save a copy of value function for the convergence criterion at the step
    #     for a in range(env._num_actions):
    #         # q[:, a] = env._rewards[:, a] + env._gamma * env._T[a].dot(v_old) # calculate Q-value
    #         q[:, a] = env._rewards[:, a] + env._gamma * T[a].dot(v_old) # calculate Q-value
    #     v = np.max(q, axis=1) # update value function

    #     if np.linalg.norm(v - v_old) < tol: # convergence criterion
    #         break

    v[NORMAL_STATE] = np.max(env._rewards[NORMAL_STATE, :])
    while True:
        v_old = np.copy(v) # save a copy of value function for the convergence criterion at the step
        for a in range(env._num_actions):
            # q[:, a] = env._rewards[:, a] + env._gamma * env._T[a].dot(v_old) # calculate Q-value
            q[:, a] = env._rewards[:, a] + env._gamma * T[a].dot(v_old) # calculate Q-value
            q[NORMAL_STATE, a] = env._rewards[NORMAL_STATE, a]
        
        v = np.max(q, axis=1) # update value function

        # diff = np.concatenate([v[:NORMAL_STATE], v[NORMAL_STATE+1:]]) - np.concatenate([v_old[:NORMAL_STATE], v_old[NORMAL_STATE+1:]])
        # if np.linalg.norm(diff) < tol: # convergence criterion
        #     break
        if np.linalg.norm(v - v_old) < tol: # convergence criterion
            break
    
    return v


def policy_evaluation(env, policy: np.ndarray, tol: float = 1e-10):
    """
    Evaluate the value function for a given policy using iterative policy evaluation.

    Parameters:
        env (WUREnvironment): The environment in which the agent is interacting.
        policy (np.ndarray): The policy to be evaluated.
        tol (float): The tolerance for convergence criterion. Defaults to 1e-10.

    Returns:
        np.ndarray: The value function for the given policy.
    """
    v = np.zeros(env._num_states) # initialize value function
    q = np.zeros((env._num_states, env._num_actions)) #initialize Q-value

    v[NORMAL_STATE] = np.max(env._rewards[NORMAL_STATE, :])
    while True:
        v_old = np.copy(v) # save a copy of value function for the convergence criterion at the step
        for a in range(env._num_actions):
            q[:, a] = env._rewards[:, a] + env._gamma * env._T[a].dot(v_old) #calculate Q-value
            q[NORMAL_STATE, a] = env._rewards[NORMAL_STATE, a]
        for s in range(env._num_states):
            if s == NORMAL_STATE:
                continue
            action_taken = policy[s] # obtain the action determined by the policy
            v[s] = q[s, action_taken] #calculate value function by $v(s) = max_a Q(s,a)$
        if np.linalg.norm(v - v_old) < tol: # convergence criterion
            break

    return v

def compute_subopt(env, agent):
    
    policy = np.array([
        agent.argmax_with_random_tie_breaking(
            agent._Q[state, :]
        ) for state in range(env._num_states)
    ])
    # policy = np.argmax(agent._Q, axis=1)

    V_opt = value_iteration(env)
    V_pi = policy_evaluation(env, policy)

    V_opt = np.concatenate([V_opt[:NORMAL_STATE], V_opt[NORMAL_STATE+1:]])
    V_pi = np.concatenate([V_pi[:NORMAL_STATE], V_pi[NORMAL_STATE+1:]])
    i = np.argmax(np.abs(V_opt - V_pi))
    return V_opt[i] - V_pi[i]