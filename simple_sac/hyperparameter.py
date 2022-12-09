import numpy as np

class hp(object):
	#trainer
    max_steps = int(1e7)
    episode_max_steps = 50
    save_model_interval = 5000
    save_summary_interval = int(1e3)
    random_action_prob = 0.1
    
    # env
    joint1_range = [-5.0, 5.0]
    joint2_range = [-5.0, 5.0]
    joint_max = [5.0, 5.0]
    joint_min = [-5.0, -5.0]
    margin = 0.0 #makes max value lower and min value higher and also increase node check boundary
    env_noise = 0.002 #noise in step fuction
    c_check_acc = 0.2 #accuracy of path collision(step check)
    state_dim = 4 #delta joint and goal
    action_dim = 2
    step_size = 0.5
    goal_bound = 0.1

    # agent
    gpu = 0
    max_action = 1.
    
    lr = 3e-4
    sigma = 0.1
    tau = 0.005

    policy_name = "SAC"
    update_interval = 1
    batch_size = 512
    discount = 0.98
    n_warmup = 2000
    n_epoch = 1
    max_grad = 10
    memory_capacity = 1000000
    
    lr_actor = 0.001
    lr_critic = 0.001
    tau = 0.005
    actor_l2 = 1.0
    her_k = 4
    max_time_step = 50

    target_noise_std = 0.2
    target_noise_clip = 0.5
    policy_delay = 2
    discount_factor = 0.98
    batch_size = 512
    memory_size = 1000000
    random_action_prob = 0.1
    action_noise_std = 0.1

	# network
    hidden_layer = 400
    LOG_SIG_CAP_MAX = 2  # np.e**2 = 7.389
    LOG_SIG_CAP_MIN = -20  # np.e**-10 = 4.540e-05
    EPS = 1e-6

