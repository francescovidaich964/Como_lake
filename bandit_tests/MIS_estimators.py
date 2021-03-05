
import numpy as np



###############################
### MIS estimator Francesco ###
###############################

def MIS_estimator(nu, thetas, rewards, t, return_type=0, beta=1):
    
    # Check if rewards and thetas have the same lenght
    assert len(thetas) == len(rewards), "Thetas and rewards have different lengths"
#     if len(thetas) != len(rewards):
#         print("\n ----- Thetas and rewards have different lengths ----- \n")
#         return None

    # Compute time of past rewards and prepare array for them
    t0 = (t+1) - len(rewards)
    timesteps = np.arange(t0, t+1)
    sum_terms = np.array([]) 
    
    # Compute constant normalization term (for beta<1)
    if beta<1:
        beta_normalization = (1-beta)/(1-beta**(len(rewards)-1))
        
    # Compute the weighted rewards inside the time window (with lenght alpha)
    for i in timesteps:
        
        # Compute importance weight given the type of the estimator (with beta ir not)
        IS_weight = nu.theta_pdf(thetas[i-t0], t+1) / nu.theta_pdf(thetas[i-t0], i)
        if beta == 1:
            IS_weight = IS_weight / len(rewards)
        else:
            IS_weight = IS_weight * beta_normalization * beta**(t-i)
        
        # IF we are interested in the weights, don't compute weighted rewards
        if return_type == 2:
            sum_terms = np.append(sum_terms, IS_weight)
        else:
            IS_reward = rewards[i-t0] * IS_weight
            sum_terms = np.append(sum_terms, IS_reward)
    
    # If we are interested in just the estimated performance, sum the weighted rewards
    if return_type == 0:
        return np.sum(sum_terms)
    else:
        return sum_terms
    
    
    
    
    
############################
### MIS estimator Pierre ###
############################
    
def MIS_estimator_other(nu, thetas, rewards, alpha=10, beta=1):
    
    # Check if rewards and thetas have the same lenght
    assert len(thetas) == len(rewards), "Thetas and rewards have different lengths"
    assert len(thetas) > alpha, "Not enough data for the estimator"

    # Compute time of past rewards and prepare array for them
    t = len(thetas)
    timesteps = np.arange(t-alpha, t)
    IS_est = []
    
    # Compute constant normalization term (for beta<1)
    if beta<1:
        beta_normalization = np.array([beta**(t-i) for i in timesteps])
        beta_normalization *= (1-beta)/(1-beta**(alpha-1))
    else: 
        beta_normalization = 1/alpha
        
        
    # Compute the weighted rewards inside the time window (with lenght alpha)
    IS_weights = np.array([nu.theta_pdf(thetas[i], t) / nu.theta_pdf(thetas[i], i) for i in timesteps])
    IS_est  = beta_normalization * IS_weights * rewards[timesteps]

    return IS_est, np.sum(IS_est), IS_weights#, rewards[timesteps], beta_normalization





#############################
### MIS estimator with BH ###
#############################

def MIS_bh_estimation(nu, thetas, rewards, alpha=10, beta=1):

    # Check if rewards and thetas have the same lenght
    assert len(thetas) == len(rewards), "Thetas and rewards have different lengths"
    assert len(thetas) >= alpha, "Not enough data for the estimator"

    # Compute time of past rewards and prepare array for them
    t = len(thetas)
    timesteps = np.arange(t-alpha, t)
    IS_est = []

    # Compute constant normalization term (for beta<1)
    if beta<1:
        beta_normalization = np.array([beta**(t-i) for i in timesteps])
        beta_normalization *= (1-beta)/(1-beta**(alpha-1))
    else: 
        beta_normalization = 1/alpha


    # Compute the weighted rewards inside the time window (with length alpha)
    MIS_beta = np.array([np.sum([nu.theta_pdf(thetas[i], k) for k in timesteps]) for i in timesteps ])
    IS_weights = np.array([nu.theta_pdf(thetas[i], t) for i in timesteps])
    IS_weights = IS_weights*MIS_beta
    IS_est  = beta_normalization * IS_weights * rewards[timesteps]

    return IS_est, np.sum(IS_est), IS_weights#, rewards[timesteps], beta_normalization




def range_IS_weight(alpha, range_len):
    ar = np.concatenate((np.ones(2*alpha+1), np.zeros(range_len+alpha)))
    roll = np.stack([np.roll(ar, shift=i) for i in range(alpha+range_len+1)])
    return roll[:,alpha:-alpha-1]
    

    
    
def MIS_bh_estimation_range(nu, thetas, rewards, range_len, alpha=10, beta=1):
    t = len(thetas)
    shift = t-range_len-alpha-1
    timesteps = np.arange(t-alpha-range_len, t).reshape(-1,1)
    range_weight = range_IS_weight(alpha, range_len)
    proba_nu = np.zeros_like(range_weight)
    
    for (i,k) in tqdm(zip(np.where(range_weight)[0],np.where(range_weight)[1])):
        proba_nu[i,k] = nu.theta_pdf(thetas[i], k) 
        
    if beta<1:
        beta_normalization = np.array([beta**(t-i) for i in timesteps])
        beta_normalization *= (1-beta)/(1-beta**(alpha-1))
    else: 
        beta_normalization = 1/alpha
        
    IS_est_all = []
    timesteps = np.arange(shift,shift+alpha)
        
    for t in range(range_len):
        MIS_beta = np.array([ np.sum(proba_nu[t:t+alpha,i]) for i in range(t,t+alpha)])
        IS_weights = proba_nu[t+alpha,t:t+alpha]
        IS_weights = IS_weights*MIS_beta
        IS_est = beta_normalization * IS_weights * rewards[timesteps+t]
        IS_est_all.append(np.sum(IS_est))
        
    return IS_est_all
    

    

########################################
### MIS estimator with BH - Variance ###
########################################
    
    
def renyi_normal(alpha, mu_1, sigma_1, mu_2, sigma_2):
    """ D_\alpha(P_1||P_2)"""
    mu_diff = (mu_1-mu_2).reshape(-1,1)
    sigma_alpha = alpha*sigma_1 + (1-alpha)*sigma_2
    renyi_div = np.linalg.det(sigma_alpha)
    renyi_div /= (np.linalg.det(sigma_1)**(1-alpha)*np.linalg.det(sigma_2)**alpha)
    renyi_div = -np.log(renyi_div)/(2*(alpha-1))
    renyi_div += alpha/2*np.matmul(mu_diff.T, np.matmul(np.linalg.invsigma_alpha), mu_diff)
    return renyi_div

def exp_2_renyi_div_normal_1d(mu_1, sigma_1, mu_2, sigma_2, alpha):
    sigma_alpha = alpha*sigma_1 + (1-alpha)*sigma_2
    renyi_div = np.exp((mu_1-mu_2)**2/sigma_alpha)
    renyi_div /= np.sqrt(sigma_alpha*sigma_1/sigma_2)
    return renyi_div
    
def MIS_bh_variance(nu, thetas, rewards, R_inf=1, alpha=10, beta=1):
    t = len(thetas)
    timesteps = np.arange(t-alpha, t)

    renyi_div = [exp_2_renyi_div_normal_1d(nu.theta_mean(t), nu.sigma_theta, nu.theta_mean(i), nu.sigma_theta, alpha) for i in timesteps]
    renyi_div = np.array(renyi_div)
    
    return R_inf*alpha/np.sum(1/renyi_div)
    
# def MIS_bh_variance_range(nu, thetas, rewards, range_len, R_inf=1, alpha=10, beta=1):
#     t = len(thetas)
#     shift = t-range_len-alpha-1
#     timesteps = np.arange(t-alpha-range_len, t).reshape(-1,1)
#     range_weight = range_IS_weight(alpha, range_len)
#     proba_nu = np.zeros_like(range_weight)
    
#     for (i,k) in tqdm(zip(np.where(range_weight)[0],np.where(range_weight)[1])):
#         proba_nu[i,k] = nu.theta_pdf(thetas[i], k) 