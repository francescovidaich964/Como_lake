
### Here we will build functions to define the different 
### performance estimators and their gradients.

import numpy as np



### TRAJECTORY REWARD ###

def Trajectory_Return(nu, thetas, rewards, t, beta=1, gamma=1)

    # Given present time t, get time of past rewards
    t0 = t-len(rewards)
    timesteps = np.arange(t-len(rewards), t+1)

    # Compute IS weight of the trajectory (product of per_step IS weights)
    IS_weight = 1
    for i in timesteps:
        IS_weight *= nu.theta_pdf(thetas[k-t0], t+1) / nu.theta_pdf(thetas[k-t0], k)
        
    # Compute the return as sum of weighted rewards
    performance = 0
    for i in timesteps:
        weighted_reward = beta**(t-i) * gamma**(i-t0) * rewards[i-t0]
        performance += weighted_reward

    # Return the product of the 2 quantities
    return IS_weight * performance




### PER-STEP REWARD ###

def PerStep_Reward_prod(nu, thetas, rewards, t, beta=1, gamma=1)

    # Given present time t, get time of past rewards
    t0 = t-len(rewards)
    timesteps = np.arange(t-len(rewards), t+1)

    # Compute and sum the weighted rewards
    performance = 0

    for i in timesteps:

        # IS weight is computed as product of per_step IS weights up to i
        IS_weight = 1
        for k in range(t0, i+1):
            IS_weight *= nu.theta_pdf(thetas[k-t0], t+1) / nu.theta_pdf(thetas[k-t0], k)
        
        # Compute weighted reward and sum it to the performance
        weighted_reward = beta**(t-i) * gamma**(i-t0) * rewards[i-t0] * IS_weight
        performance += weighted_reward

    return performance




### PER-STEP REWARD (no product) ###

def PerStep_Reward(nu, thetas, rewards, t, beta=1, gamma=1)

    # Given present time t, get time of past rewards
    t0 = t-len(rewards)
    timesteps = np.arange(t-len(rewards), t+1)

    # Compute and sum the weighted rewards
    performance = 0

    for i in timesteps:
        IS_weight = nu.theta_pdf(thetas[i-t0], t+1) / nu.theta_pdf(thetas[i-t0], i)
        weighted_reward = beta**(t-i) * gamma**(i-t0) * rewards[i-t0] * IS_weight
        performance += weighted_reward

    return performance