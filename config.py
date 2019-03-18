import tensorflow as tf
import time

class config_lander:
    def __init__(self, use_baseline, use_critic, use_attention, use_mask):
        self.env_name ="LunarLander-v2"
        self.record = False
        baseline_str = 'baseline' if use_baseline else 'no_baseline'
        attention_str = 'attention' if use_attention else 'control'

        # gradient clipping
        self.grad_clip = True
        self.clip_val = 5

        # output config
        self.use_mask = use_mask
        self.output_path = "results/{}-{}-{}/".format(attention_str, time.time(), self.env_name)
        self.model_output = self.output_path + "model.weights/"
        self.log_path = self.output_path + "log.txt"
        self.plot_output = self.output_path + "scores.png"
        self.record_path = self.output_path
        self.record_freq = 5
        self.summary_freq = 1

        # model and training config
        self.num_batches = 200  # number of batches trained on
        self.batch_size = 300  # number of steps used to compute each policy update
        self.max_ep_len = 300  # maximum episode length
        self.learning_rate = 1e-4
        self.gamma = 0.99  # the discount factor

        # model architecture config
        self.use_baseline = use_baseline
        self.use_critic = use_critic
        self.use_attention = use_attention
        self.normalize_advantage = True
        self.q_K = 1

        # parameters for the attention model
        self.attn_n_layers = 10
        self.attn_n_layer_size = 128
        self.attn_activation = tf.nn.relu

        # parameters for the policy and baseline models
        self.n_layers = 2
        self.n_layers += self.attn_n_layers if not use_attention else 0
        self.layer_size = 128
        self.activation = tf.nn.relu

        # parameters for replay buffers
        self.memory_len = 10
        self.percolate_len = 2

        # since we start new episodes for each batch
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size

class config_cartpole:
    def __init__(self, use_baseline):
        self.env_name="CartPole-v0"
        self.record = False
        baseline_str = 'baseline' if use_baseline else 'no_baseline'

        # output config
        self.output_path = "results/{}-{}/".format(self.env_name, baseline_str)
        self.model_output = self.output_path + "model.weights/"
        self.log_path     = self.output_path + "log.txt"
        self.plot_output  = self.output_path + "scores.png"
        self.record_path  = self.output_path
        self.record_freq = 5
        self.summary_freq = 1

        # model and training config
        self.num_batches            = 100 # number of batches trained on
        self.batch_size             = 1000 # number of steps used to compute each policy update
        self.max_ep_len             = 1000 # maximum episode length
        self.learning_rate          = 3e-2
        self.gamma                  = 1.0 # the discount factor
        self.use_baseline           = use_baseline
        self.normalize_advantage    = True

        # parameters for the policy and baseline models
        self.n_layers               = 1
        self.layer_size             = 16
        self.activation             = tf.nn.relu

        # since we start new episodes for each batch
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size

class config_pendulum:
    def __init__(self, use_baseline):
        self.env_name="InvertedPendulum-v1"
        self.record = False
        baseline_str = 'baseline' if use_baseline else 'no_baseline'

        # output config
        self.output_path = "results/{}-{}/".format(self.env_name, baseline_str)
        self.model_output = self.output_path + "model.weights/"
        self.log_path     = self.output_path + "log.txt"
        self.plot_output  = self.output_path + "scores.png"
        self.record_path  = self.output_path
        self.record_freq = 5
        self.summary_freq = 1

        # model and training config
        self.num_batches            = 100 # number of batches trained on
        self.batch_size             = 1000 # number of steps used to compute each policy update
        self.max_ep_len             = 1000 # maximum episode length
        self.learning_rate          = 3e-2
        self.gamma                  = 1.0 # the discount factor
        self.use_baseline           = use_baseline
        self.normalize_advantage    = True

        # parameters for the policy and baseline models
        self.n_layers               = 1
        self.layer_size             = 16
        self.activation             = tf.nn.relu

        # since we start new episodes for each batch
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size

class config_cheetah:
    def __init__(self, use_baseline):
        self.env_name="HalfCheetah-v1"
        self.record = False
        baseline_str = 'baseline' if use_baseline else 'no_baseline'

        # output config
        self.output_path = "results/{}-{}/".format(self.env_name, baseline_str)
        self.model_output = self.output_path + "model.weights/"
        self.log_path     = self.output_path + "log.txt"
        self.plot_output  = self.output_path + "scores.png"
        self.record_path  = self.output_path
        self.record_freq  = 5
        self.summary_freq = 1

        # gradient clipping
        grad_clip = True
        clip_val = 5

        # model and training config
        self.num_batches            = 100 # number of batches trained on
        self.batch_size             = 50000 # number of steps used to compute each policy update
        self.max_ep_len             = 1000 # maximum episode length
        self.learning_rate          = 3e-2
        self.gamma                  = 0.9 # the discount factor

        # model architecture config
        self.use_baseline = use_baseline
        self.use_critic = use_critic
        self.use_attention = use_attention
        self.normalize_advantage    = True

        # parameters for the policy and baseline models
        self.n_layers               = 3
        self.layer_size             = 32
        self.activation             = tf.nn.relu
        # since we start new episodes for each batch
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size

def get_config(env_name,
               use_baseline=False,
               use_critic=False,
               use_attention=False,
               use_mask=False):
    if env_name == 'cartpole':
        return config_cartpole(use_baseline, use_critic, use_attention, use_mask)
    elif env_name == 'pendulum':
        return config_pendulum(use_baseline, use_critic, use_attention, use_mask)
    elif env_name == 'cheetah':
        return config_cheetah(use_baseline, use_critic, use_attention, use_mask)
    elif env_name == 'lander':
        return config_lander(use_baseline, use_critic, use_attention, use_mask)
    elif env_name == 'lander-continuous':
        config = config_lander(use_baseline, use_critic, use_attention, use_mask)
        config.env_name = 'LunarLanderContinuous-v2'
        return config
