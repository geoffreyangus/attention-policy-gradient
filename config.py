import tensorflow as tf

class config_cartpole:
    def __init__(self, use_baseline):
        self.env_name="CartPole-v0"
        self.record = True
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
        self.record = True
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
        self.record = True
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
        self.batch_size             = 50000 # number of steps used to compute each policy update
        self.max_ep_len             = 1000 # maximum episode length
        self.learning_rate          = 3e-2
        self.gamma                  = 0.9 # the discount factor
        self.use_baseline           = use_baseline
        self.normalize_advantage    = True

        # parameters for the policy and baseline models
        self.n_layers               = 3
        self.layer_size             = 32
        self.activation             = tf.nn.relu

        # since we start new episodes for each batch
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size


class config_pong:
    def __init__(self, use_baseline):
        # env config
        self.render_train = False
        self.render_test = False
        self.env_name = "Pong-v0"
        self.overwrite_render = True
        self.record = True
        self.high = 255.

        # output config
        self.output_path = "results/attention/"
        self.model_output = self.output_path + "model.weights/"
        self.log_path = self.output_path + "log.txt"
        self.plot_output = self.output_path + "scores.png"
        self.record_path = self.output_path + "monitor/"

        # model and training config
        self.num_episodes_test = 50
        self.grad_clip = True
        self.clip_val = 10
        self.saving_freq = 250000
        self.log_freq = 50
        self.eval_freq = 250000
        self.record_freq = 250000
        self.soft_epsilon = 0.05

        # my hyperparams
        self.n_layers = 1
        self.layer_size = 16
        self.activation = tf.nn.relu
        self.learning_rate = 1e-3
        self.use_baseline = use_baseline

        self.skip_frame = 4



def get_config(env_name, baseline):
    if env_name == 'cartpole':
        return config_cartpole(baseline)
    elif env_name == 'pendulum':
        return config_pendulum(baseline)
    elif env_name == 'cheetah':
        return config_cheetah(baseline)
    elif env_name == 'pong':
        return config_pong(baseline)
