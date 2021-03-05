
import numpy as np
import scipy.stats

# Define sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))



#######################
###   ENVIRONMENT   ###
#######################

class environment:
    
    ### Define the environment: NS_context=True  -> non-stat context distribution    
    ###                         NS_context=False -> non-stat reward function
    ###                         remove_contexts -> x_t=0 always, reward function becomes non-stat
    def __init__(self, NS_context, remove_context=False, sigma_x=1, t0=0, **kwargs):
        
        # Store the given parameters
        self.t = t0
        self.t_init = t0
        self.sigma_x = sigma_x
        self.remove_context = remove_context
        
        # If we don't want to use context, use a non-stat reward function
        if remove_context:
            self.NS_context = False
        else:
            self.NS_context = NS_context
        
        # if params that define the mean are not given, take random ones
        self.var_names = ['A', 'phi', 'psi'] 
        for var in self.var_names:
            if var in kwargs.keys():
                setattr(self, var, kwargs[var])
            else:
                setattr(self, var, np.random.rand())
                
    def reset(self):
        self.t = self.t_init
    

    
    ### Definition of the non-stationary process: 
    ###    If NS_context = True,  it defines the mean of the context distribution
    ###    If NS_context = False, it defines a term in the reward function
    def non_stat_process(self, t):
        return self.A * np.sin(self.phi*t + self.psi)
        
    
    
    ### Compute the context mean at present timestep 'self.t'. If context distr is 
    ### stationary or if x_t is fixed to 0, the mean is always equal to 0
    def x_mean(self, t=None):
        if self.remove_context or not(self.NS_context):
            return 0
        else:
            return self.non_stat_process(self.t)

        
    ### Sample a context from the distibution at time 'self.t'
    def sample_x(self):
        if self.remove_context:
            return 0
        else:
            return scipy.stats.norm.rvs(loc   = self.x_mean(), 
                                        scale = self.sigma_x)
    
    
    
    
    ### Sample reward from a Bernoulli distribution that depends on the action and the context; 
    ### The parameter is computed differently if we have non-stat contexts or non-stat rewards
    def get_reward(self, x, action, get_p=False):
        
        # Compute bernoulli parameter for the correct case
        if self.NS_context:
            p_t = sigmoid( x - self.x_mean() )
        else:
            #initial_reward = self.non_stat_process(self.t)                  #      ----  ATTENTION  ----
            #p_t = sigmoid( x + initial_reward )
            p_t = self.non_stat_process(self.t) + 0.5
        
        # If we want to study the distributions, return parameter instead of reward
        if get_p:
            if action:
                return p_t
            else:
                return 1-p_t
        
        # Sample a reward given the chosen action
        if action:
            return scipy.stats.bernoulli.rvs(p_t)
        else:
            return scipy.stats.bernoulli.rvs(1 - p_t)
            
            
    
    
    ### Use the hyperpolicy nu to perform many subsequent steps
    ### Returns the evolution of many variables
    def play(self, nu, n_steps, get_p=False):
        
        contexts = np.array([])
        thetas   = np.array([])
        rewards  = np.array([])
        actions  = np.array([])
        non_stat_process = np.array([])
        theta_means      = np.array([])
    
        # At each timestep...
        for i in range(n_steps):
            
            # Sample a policy from the hyperpolicy 
            # (given t and the tipe of non-stationarity)
            policy = nu.sample_policy(self.t, self.NS_context)
            
            # Sample a context from the corresponding distribution
            x_t = self.sample_x()
            
            # Perform a step sampling the action from current policy 
            action = policy.action(x_t)
            reward = self.get_reward(x_t, action, get_p)
            
            # Store current values in the arrays
            contexts = np.append(contexts, x_t)
            thetas   = np.append(thetas, policy.theta)
            rewards  = np.append(rewards, reward)
            actions  = np.append(actions, action)
            non_stat_process = np.append(non_stat_process, self.non_stat_process(self.t))
            theta_means   = np.append(theta_means, nu.theta_mean(self.t))
            
            # Update time variable
            self.t += 1

        return contexts, thetas, rewards, actions, non_stat_process, theta_means
    
    def play_deter(self, nu, n_steps, get_p=False):
        
        contexts = np.array([])
        thetas   = np.array([])
        rewards  = np.array([])
        actions  = np.array([])
        non_stat_process = np.array([])
        theta_means      = np.array([])
    
        # At each timestep...
        for i in range(n_steps):
            
            # Sample a policy from the hyperpolicy 
            # (given t and the tipe of non-stationarity)
            policy = policy_class(nu.theta_mean(self.t), self.NS_context, False)
            
            # Sample a context from the corresponding distribution
            x_t = self.sample_x()
            
            # Perform a step sampling the action from current policy 
            action = policy.action(x_t)
            reward = self.get_reward(x_t, action, get_p)
            
            # Store current values in the arrays
            contexts = np.append(contexts, x_t)
            thetas   = np.append(thetas, policy.theta)
            rewards  = np.append(rewards, reward)
            actions  = np.append(actions, action)
            non_stat_process = np.append(non_stat_process, self.non_stat_process(self.t))
            theta_means   = np.append(theta_means, nu.theta_mean(self.t))
            
            # Update time variable
            self.t += 1

        return contexts, thetas, rewards, actions, non_stat_process, theta_means
    
    
    
    
########################
###   POLICY CLASS   ###
########################
    
class policy_class:
    
    ### Theta is the only param and reprsents an estimate of the non-stat 
    ### process of the environment (that represent different )
    def __init__(self, theta, NS_context, stochastic=True):
        self.theta = theta
        self.stochastic = stochastic
        self.NS_context = NS_context
    
    ### Return the bernoulli distr over the actions given the context
    ### (makes sense only if policy is stochastic, i.e. action-based case)
    def bernoulli_distr(self, x):
        if self.NS_context:
            p = sigmoid(x - self.theta)
        else:
            p = sigmoid(x + self.theta)
        return (1-p,p)
        
     
    ### Choose action given the sampled context (and depending on the problem)
    def action(self, x):
        
        ### IF env has nonstationary Context distribution,
        ### action = 1 has higher prob (or chosen) if x-theta > 0
        if self.NS_context:
            if self.stochastic:
                p = sigmoid(x - self.theta)
                return scipy.stats.bernoulli.rvs(p)
            else:
                return (x - self.theta > 0)
            
        ### IF env has nonstationary reward function,
        ### action = 1 has higher prob (or chosen) if x+theta > 0
        else:
            if self.stochastic:
                p = sigmoid(x + self.theta)
                return scipy.stats.bernoulli.rvs(p)
            else:
                return (x + self.theta > 0)
            
            
      
    

    
    
#######################
###   HYPER-POLICY  ###
#######################
    
class hyperpolicy:
        
    def __init__(self, param_based=True, sigma_theta=1, theta_means=None, A=None, phi=None, psi=None):
        
        # Store the given parameters
        self.sigma_theta = sigma_theta
        self.stochastic = param_based
        
        # If custom NS_process is given, define theta means with that
        # else, use params (or generate random ones) to build the process
        if theta_means is not None:
            self.theta_mean_values = theta_means
        else:
            self.A = A or 5*np.random.rand()
            self.phi = phi or np.random.rand()
            self.psi = psi or 2*np.pi*np.random.rand()
               
    
    # If hyperpolicy has not an already defined NS process, build it with params
    def theta_mean(self, t):
        if hasattr(self, 'theta_mean_values'):
            return self.theta_mean_values[t]
        else:
            return self.A * np.sin(self.phi*t + self.psi)
    
        
    def theta_pdf(self, theta, t):
        return scipy.stats.norm.pdf(theta, loc=self.theta_mean(t), scale=self.sigma_theta)
    
    
    def sample_theta(self, t):
        if self.stochastic:
            return scipy.stats.norm.rvs(loc=self.theta_mean(t), scale=self.sigma_theta)
        else:
            return self.theta_mean(t)

    def sample_policy(self, t, NS_context=False):
        theta = self.sample_theta(t)
        return policy_class(theta, NS_context, not(self.stochastic) )
    
    
    #def update_params(self, delta_params):
        
        # TO DO
        