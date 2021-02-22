
### Here we will build functions to define the different 
### performance estimators and their gradients.

import numpy as np



### TRAJECTORY REWARD ###

def Trajectory_Return(nu, thetas, rewards, t, beta=1, gamma=1):

    # Check if rewards and thetas have the same lenght
    if len(thetas) != len(rewards):
        print("\n ----- Thetas and rewards have different lengths ----- \n")
        return None

    # Given present time t, get time of past rewards
    t0 = (t+1) - len(rewards)
    timesteps = np.arange(t0, t+1)
    
    
    # Compute IS weight of the trajectory (product of per_step IS weights)
    IS_weight = 1
    for i in timesteps:
        IS_weight *= nu.theta_pdf(thetas[i-t0], t+1) / nu.theta_pdf(thetas[i-t0], i)

    # Compute the return as sum of weighted rewards
    performance = 0
    for i in timesteps:
        weighted_reward = beta**(t-i) * gamma**(i-t0) * rewards[i-t0]
        performance += weighted_reward

    # Return the product of the 2 quantities
    return IS_weight * performance




### PER-STEP REWARD ###

def PerStep_Reward_prod(nu, thetas, rewards, t, beta=1, gamma=1):

    # Check if rewards and thetas have the same lenght
    if len(thetas) != len(rewards):
        print("\n ----- Thetas and rewards have different lengths ----- \n")
        return None

    # Given present time t, get time of past rewards
    t0 = (t+1) - len(rewards)
    timesteps = np.arange(t0, t+1)

    # Compute and sum the weighted rewards
    performance = 0

    for i in timesteps:

        # IS weight is computed as product of per_step IS weights up to i
        if i == t0:
            IS_weight = nu.theta_pdf(thetas[i-t0], t+1) / nu.theta_pdf(thetas[i-t0], i)
        else:
            IS_weight *= nu.theta_pdf(thetas[i-t0], t+1) / nu.theta_pdf(thetas[i-t0], i)
        
        # Compute weighted reward and sum it to the performance
        weighted_reward = beta**(t-i) * gamma**(i-t0) * rewards[i-t0] * IS_weight
        performance += weighted_reward

    return performance




### PER-STEP REWARD (no product) ###

def PerStep_Reward(nu, thetas, rewards, t, beta=1, gamma=1):

    # Check if rewards and thetas have the same lenght
    if len(thetas) != len(rewards):
        print("\n ----- Thetas and rewards have different lengths ----- \n")
        return None

    # Given present time t, get time of past rewards
    t0 = (t+1) - len(rewards)
    timesteps = np.arange(t0, t+1)
    
    # Compute and sum the weighted rewards
    performance = 0

    for i in timesteps:
        IS_weight = nu.theta_pdf(thetas[i-t0], t+1) / nu.theta_pdf(thetas[i-t0], i)
        weighted_reward = beta**(t-i) * gamma**(i-t0) * rewards[i-t0] * IS_weight
        performance += weighted_reward

    return performance