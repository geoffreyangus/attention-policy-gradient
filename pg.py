# -*- coding: UTF-8 -*-

import os
import argparse
import sys
import logging
import time

import numpy as np
import tensorflow as tf
import gym
import scipy.signal
import inspect
from tqdm import tqdm


from utils.general import (
  get_logger,
  Progbar,
  export_plot,
  mask_obs,
  get_obs_dims,
  get_timing_signal
)
from utils.wrappers import PreproWrapper, MaxAndSkipEnv
from utils.preprocess import greyscale
from config import get_config
from utils.buffer import ReplayBuffer

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', required=True, type=str,
                    choices=['cartpole', 'pendulum', 'cheetah', 'lander'])
parser.add_argument('--baseline', dest='use_baseline', action='store_true')
parser.add_argument('--no-baseline', dest='use_baseline', action='store_false')
parser.add_argument('--mask', dest='use_mask', action='store_true')
parser.add_argument('--no-mask', dest='use_mask', action='store_false')
parser.add_argument('--attention', dest='use_attention', action='store_true')
parser.add_argument('--no-attention', dest='use_attention', action='store_false')
parser.add_argument('--critic', dest='use_critic', action='store_true')
parser.add_argument('--no-critic', dest='use_critic',action='store_false')
parser.set_defaults(use_baseline=True, use_mask=False, use_attention=False, use_critic=False)

def build_attention_layer(
  observation,
  memories,
  scope,
  n_layers,
  size
  ):
  """
  Build a network that attends to memories given observation.

  Args:
    observation (np.matrix(B, |state|))
    memories (np.matrix(B, T, |state| * 2 + |actions| + 2))
  """
  observation = tf.expand_dims(observation, 1) # add time dimension
  D_obs = observation.get_shape().as_list()[2]
  B, T, D_mem = memories.get_shape().as_list()

  Q_mem = memories
  Q_flat = tf.layers.Flatten()(Q_mem)
  with tf.variable_scope(scope):
    for i in range(n_layers):
      Q_flat = tf.layers.dense(Q_flat, size, activation=tf.nn.relu)
    # create |state| embedding
    Q_mem = tf.layers.dense(Q_mem, D_mem, activation=tf.nn.relu)
    Q_obs = tf.layers.dense(Q_mem, D_obs)
    # weights will thus have shape (B, T, 1)
    attention_weights = tf.matmul(Q_obs, observation, transpose_b=True)
    attention_weights = tf.nn.softmax(attention_weights, axis=1)
    # context vector will have shape (B, |state|)
    C = tf.matmul(attention_weights, Q_obs, transpose_a=True)
    C = tf.squeeze(C, axis=1)
    C = tf.layers.dense(C, size)
    masked_feature_logit = tf.layers.dense(Q_flat, 1)
    return C, Q_mem, masked_feature_logit


def build_mlp(
  mlp_input,
  output_size,
  scope,
  n_layers,
  size,
  output_activation=None):
  """
  Build a feed forward network (multi-layer perceptron, or mlp)
  with 'n_layers' hidden layers, each of size 'size' units.
  Use tf.nn.relu nonlinearity between layers.
  Args:
          mlp_input: the input to the multi-layer perceptron
          output_size: the output layer size
          scope: the scope of the neural network
          n_layers: the number of hidden layers of the network
          size: the size of each layer:
          output_activation: the activation of output layer
  Returns:
          The tensor output of the network

  TODO: Implement this function. This will be similar to the linear
  model you implemented for Assignment 2.
  "tf.layers.dense" and "tf.variable_scope" may be helpful.

  A network with n hidden layers has n 'linear transform + nonlinearity'
  operations followed by the final linear transform for the output layer
  (followed by the output activation, if it is not None).

  """
  x = mlp_input
  with tf.variable_scope(scope):
    for i in range(n_layers):
      x = tf.layers.dense(x, size, activation=tf.nn.relu)
    output = tf.layers.dense(x, output_size, output_activation)
  return output


class PG(object):
  """
  Abstract Class for implementing a Policy Gradient Based Algorithm
  """

  def __init__(self, env, config, logger=None):
    """
    Initialize Policy Gradient Class

    Args:
            env: an OpenAI Gym environment
            config: class with hyperparameters
            use_mask: train time, omit velocity features in state
            logger: logger instance from the logging module

    You do not need to implement anything in this function. However,
    you will need to use self.discrete, self.observation_dim,
    self.action_dim, and self.lr in other methods.

    """
    # directory for training outputs
    if not os.path.exists(config.output_path):
      os.makedirs(config.output_path)

    # store hyperparameters
    self.config = config
    if self.config.use_mask:
      print('Using mask...')
    self.logger = logger
    if logger is None:
      self.logger = get_logger(config.log_path)
    self.env = env

    # discrete vs continuous action space
    self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
    self.observation_dim = get_obs_dims(self.config.env_name, self.config.use_mask)
    self.action_dim = self.env.action_space.n if self.discrete else self.env.action_space.shape[0]
    self.lr = self.config.learning_rate

    # for milestone: capture raw tuple embedding
    self.memory_dim = 6 #self.observation_dim * 2 + self.action_dim + 1 + 1 # (s, a, r, s', done_mask)
    self.replay_buffer = ReplayBuffer(self.config.memory_len + 1, 1, action_dim=self.action_dim)
    self.percolated_buffer = ReplayBuffer(self.config.percolate_len + 1, 1, action_dim=self.action_dim)

    # build model
    self.build()

  def add_placeholders_op(self):
    """
    Add placeholders for observation, action, and advantage:
        self.observation_placeholder, type: tf.float32
        self.action_placeholder, type: depends on the self.discrete
        self.advantage_placeholder, type: tf.float32

    HINT: Check self.observation_dim and self.action_dim
    HINT: In the case of continuous action space, an action will be specified by
    'self.action_dim' float32 numbers (i.e. a vector with size 'self.action_dim')
    """
    self.observation_placeholder = tf.placeholder(tf.float32, shape=(None, self.observation_dim), name="observation")
    if self.discrete:
      self.action_placeholder = tf.placeholder(tf.int32, shape=(None), name="action_discrete")
    else:
      self.action_placeholder = tf.placeholder(tf.float32, shape=(None, self.action_dim), name="action_continuous")

    # Define a placeholder for advantages
    self.advantage_placeholder = tf.placeholder(tf.float32, shape=(None), name="advantages")

    self.memory_placeholder = tf.placeholder(tf.float32, shape=(None, self.config.memory_len, self.memory_dim), name="memory")
    self.percolate_placeholder = tf.placeholder(tf.float32, shape=(None, self.config.percolate_len, self.memory_dim), name="percolate")

  def build_policy_network_op(self, scope="policy_network"):
    """
    Build the policy network, construct the tensorflow operation to sample
    actions from the policy network outputs, and compute the log probabilities
    of the actions taken (for computing the loss later). These operations are
    stored in self.sampled_action and self.logprob. Must handle both settings
    of self.discrete.

    Args:
            scope: the scope of the neural network

    TODO:
    Discrete case:
        action_logits: the logits for each action
            HINT: use build_mlp, check self.config for layer_size and
            n_layers
        self.sampled_action: sample from these logits
            HINT: use tf.multinomial + tf.squeeze
        self.logprob: compute the log probabilities of the taken actions
            HINT: 1. tf.nn.sparse_softmax_cross_entropy_with_logits computes
                     the *negative* log probabilities of labels, given logits.
                  2. taken actions are different than sampled actions!

    Continuous case:
        To build a policy in a continuous action space domain, we will have the
        model output the means of each action dimension, and then sample from
        a multivariate normal distribution with these means and trainable standard
        deviation.

        That is, the action a_t ~ N( mu(o_t), sigma)
        where mu(o_t) is the network that outputs the means for each action
        dimension, and sigma is a trainable variable for the standard deviations.
        N here is a multivariate gaussian distribution with the given parameters.

        action_means: the predicted means for each action dimension.
            HINT: use build_mlp, check self.config for layer_size and
            n_layers
        log_std: a trainable variable for the log standard deviations.
            HINT: think about why we use log std as the trainable variable instead of std
            HINT: use tf.get_variables
        self.sampled_actions: sample from the gaussian distribution as described above
            HINT: use tf.random_normal
            HINT: use re-parametrization to obtain N(mu, sigma) from N(0, 1)
        self.lobprob: the log probabilities of the taken actions
            HINT: use tf.contrib.distributions.MultivariateNormalDiag

    """
    with tf.variable_scope(scope):
      # optional attention layer
      if self.config.use_attention:
        self.attn_logits = []
        self.Q_mem = []
        for head in range(1):
          attn_logits, Q_mem, self.masked_feature_logit = build_attention_layer(
              self.observation_placeholder,
              self.memory_placeholder,
              f"{scope}_attn_{head}",
              self.config.attn_n_layers,
              self.config.attn_n_layer_size
          )
          self.attn_logits.append(attn_logits)
          self.Q_mem.append(Q_mem)

        # perc_logits, _ = build_attention_layer(
        #     self.observation_placeholder,
        #     self.percolate_placeholder,
        #     f"{scope}_perc",
        #     self.config.attn_n_layers,
        #     self.config.attn_n_layer_size
        # )
        self.Q_mem = tf.concat(self.Q_mem, axis=1)
        self.attn_logits = tf.concat(self.attn_logits, axis=1)
        self.inputs = tf.concat(
            [self.observation_placeholder, self.masked_feature_logit], axis=1)
      else:
        self.Q_mem = self.observation_placeholder
        self.inputs = self.observation_placeholder
      # actor network
      if self.discrete:
        actions = build_mlp(
          self.inputs, self.action_dim, scope,
          self.config.n_layers, self.config.layer_size
        )
        self.sampled_action = tf.squeeze(tf.multinomial(actions, 1), axis=1)
        self.logprob = -1 * tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=self.action_placeholder,
          logits=actions
        )
      else:
        actions = build_mlp(
          self.inputs, self.action_dim, scope,
          self.config.n_layers, self.config.layer_size
        )
        log_std = tf.get_variable('log_std', shape=(self.action_dim), dtype=tf.float32)
        std = tf.exp(log_std)
        self.sampled_action = actions + std * tf.random_normal([self.action_dim])
        self.logprob = tf.contrib.distributions.MultivariateNormalDiag(
          actions, scale_diag=std).log_prob(self.action_placeholder)

  def add_critic_op(self, q_scope='critic', target_q_scope='target'):
      # critic network
      with tf.variable_scope(q_scope):
        self.q = tf.squeeze(build_mlp(
          self.inputs, (1 if not self.discrete else self.action_dim), q_scope,
          self.config.n_layers, self.config.layer_size
        ))
      self.q_targets = []
      self.q_k_placeholder = tf.placeholder(tf.uint8, shape=(None), name="q_k")
      for k in range(self.config.q_K):
        with tf.variable_scope(target_q_scope + f'_{k}'):
          self.q_targets.append(build_mlp(
            self.inputs, (1 if not self.discrete else self.action_dim), target_q_scope,
            self.config.n_layers, self.config.layer_size
          ))
      self.q_target_placeholder = tf.placeholder(tf.float32, shape=(None), name="q_reward")
      self.td_error = (
        self.q_target_placeholder - tf.slice(self.q, self.action_placeholder, [1])
      )
      self.loss_critic = tf.square(self.td_error)

      scope_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, q_scope)
      optim = tf.train.AdamOptimizer(self.lr * 5)

      grads_and_vars = optim.compute_gradients(self.loss_critic, scope_vars)
      if self.config.grad_clip:
        grads_and_vars = [(tf.clip_by_norm(grad, self.config.clip_val), var)
          for grad, var in grads_and_vars]

      self.train_critic_op = optim.apply_gradients(grads_and_vars)
      self.add_update_target_op()

  def add_update_target_op(self, q_scope='critic', target_q_scope='target'):
      q_scope_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, q_scope)
      target_q_scope_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, target_q_scope + f'_{self.q_k_placeholder}')

      assign_ops = []
      for i in range(len(target_q_scope_vars)):
          assign_ops.append(tf.assign(target_q_scope_vars[i], q_scope_vars[i]))
      self.update_target_op = tf.group(*assign_ops)

  def add_loss_op(self):
    """
    Compute the loss, averaged for a given batch.

    Recall the update for REINFORCE with advantage:
    θ = θ + α ∇_θ log π_θ(a_t|s_t) A_t
    Think about how to express this update as minimizing a
    loss (so that tensorflow will do the gradient computations
    for you).

    You only have to reference fields of 'self' that have already
    been set in the previous methods.

    """
    self.loss_actor = -1 * tf.reduce_mean(self.logprob * self.advantage_placeholder)

  def add_optimizer_op(self):
    """
    Set 'self.train_op' using AdamOptimizer
    HINT: Use self.lr, and minimize self.loss
    """
    # attention pretrain network
    self.masked_feature_placeholder = tf.placeholder(tf.float32, shape=(None), name="masked_target")
    self.pretrain_loss = tf.losses.mean_squared_error(self.masked_feature_placeholder, self.masked_feature_logit)
    self.pretrain_op = tf.train.AdamOptimizer(self.lr).minimize(self.pretrain_loss)

    # actor network
    self.train_op = tf.train.AdamOptimizer(self.lr).minimize(
      self.loss_actor, var_list=tf.trainable_variables(scope='policy_network/policy_network/'))

  def add_baseline_op(self, scope = "baseline"):
    """
    Build the baseline network within the scope.

    In this function we will build the baseline network.
    Use build_mlp with the same parameters as the policy network to
    get the baseline estimate. You also have to setup a target
    placeholder and an update operation so the baseline can be trained.

    Args:
        scope: the scope of the baseline network

    TODO: Set the following fields
        self.baseline
            HINT: use build_mlp, the network is the same as policy network
            check self.config for n_layers and layer_size
            HINT: tf.squeeze might be helpful
        self.baseline_target_placeholder
        self.update_baseline_op
            HINT: first construct a loss using tf.losses.mean_squared_error.
            HINT: use AdamOptimizer with self.lr

    """
    with tf.variable_scope(scope):
      self.baseline = tf.squeeze(build_mlp(
        self.observation_placeholder, 1, scope, self.config.n_layers,
        self.config.layer_size
      ), axis=1)
      self.baseline_target_placeholder = tf.placeholder(tf.float32, shape=(None), name="baseline_target")
      loss = tf.losses.mean_squared_error(self.baseline_target_placeholder, self.baseline)
      optim = tf.train.AdamOptimizer(self.lr)
      self.update_baseline_op = optim.minimize(loss)

  def build(self):
    """
    Build the model by adding all necessary variables.

    You don't have to change anything here - we are just calling
    all the operations you already defined above to build the tensorflow graph.
    """

    # add placeholders
    self.add_placeholders_op()
    # create policy net
    self.build_policy_network_op()
    # add square loss
    self.add_loss_op()
    # add optmizer for the main networks
    self.add_optimizer_op()

    # add baseline
    if self.config.use_baseline:
      self.add_baseline_op()

    if self.config.use_critic:
      self.add_critic_op()

  def initialize(self):
    """
    Assumes the graph has been constructed (have called self.build())
    Creates a tf Session and run initializer of variables

    You don't have to change or use anything here.
    """
    # create tf session
    self.sess = tf.Session()
    # tensorboard stuff
    self.add_summary()
    # initiliaze all variables
    init = tf.global_variables_initializer()
    self.sess.run(init)

    if self.config.use_critic:
      self.sess.run(self.update_target_op)

  def add_summary(self):
    """
    Tensorboard stuff.

    You don't have to change or use anything here.
    """
    # extra placeholders to log stuff from python
    self.avg_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_reward")
    self.max_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="max_reward")
    self.std_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="std_reward")

    self.eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_reward")

    # extra summaries from python -> placeholders
    tf.summary.scalar("Avg Reward", self.avg_reward_placeholder)
    tf.summary.scalar("Max Reward", self.max_reward_placeholder)
    tf.summary.scalar("Std Reward", self.std_reward_placeholder)
    tf.summary.scalar("Eval Reward", self.eval_reward_placeholder)

    # logging
    self.merged = tf.summary.merge_all()
    self.file_writer = tf.summary.FileWriter(self.config.output_path,self.sess.graph)

  def init_averages(self):
    """
    Defines extra attributes for tensorboard.

    You don't have to change or use anything here.
    """
    self.avg_reward = 0.
    self.max_reward = 0.
    self.std_reward = 0.
    self.eval_reward = 0.

  def update_averages(self, rewards, scores_eval):
    """
    Update the averages.

    You don't have to change or use anything here.

    Args:
        rewards: deque
        scores_eval: list
    """
    self.avg_reward = np.mean(rewards)
    self.max_reward = np.max(rewards)
    self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

    if len(scores_eval) > 0:
      self.eval_reward = scores_eval[-1]

  def record_summary(self, t):
    """
    Add summary to tensorboard

    You don't have to change or use anything here.
    """

    fd = {
      self.avg_reward_placeholder: self.avg_reward,
      self.max_reward_placeholder: self.max_reward,
      self.std_reward_placeholder: self.std_reward,
      self.eval_reward_placeholder: self.eval_reward,
    }
    summary = self.sess.run(self.merged, feed_dict=fd)
    # tensorboard stuff
    self.file_writer.add_summary(summary, t)

  def sample_path(self, env, num_episodes=None, pretraining=False):
    """
    Sample paths (trajectories) from the environment. Stores paths in
    replay_buffer if replay_buffer is not None.

    Args:
        num_episodes: the number of episodes to be sampled
            if none, sample one batch (size indicated by config file)
      env: open AI Gym envinronment

    Returns:
        paths: a list of paths. Each path in paths is a dictionary with
            path["observation"] a numpy array of ordered observations in the path
            path["actions"] a numpy array of the corresponding actions in the path
            path["reward"] a numpy array of the corresponding rewards in the path
        total_rewards: the sum of all rewards encountered during this "path"

    You do not have to implement anything in this function, but you will need to
    understand what it returns, and it is worthwhile to look over the code
    just so you understand how we are taking actions in the environment
    and generating batches to train on.
    """
    episode = 0
    episode_rewards = []
    paths = []
    t = 0

    while (num_episodes or t < self.config.batch_size):
      self.replay_buffer = ReplayBuffer(self.config.memory_len + 1, 1, action_dim=self.action_dim)
      state = env.reset()
      if self.config.use_mask:
        state = mask_obs(self.config.env_name, state)
      masked_feature = state[2]

      states, actions, rewards, memories, percolates, losses = [], [], [], [], [], []
      episode_reward = 0
      for step in range(self.config.max_ep_len):
        states.append(state)
        # for milestone: randomly get memory_len sets of samples
        if self.replay_buffer.can_sample(self.config.memory_len):
          memory = [self.replay_buffer.sample_stack(self.config.memory_len)[:,:self.memory_dim]]
          # tensor of shape (batch_size, self.memory_len, num_features)
          memory = np.stack(memory)
        else:
          memory = np.zeros((1, self.config.memory_len, self.memory_dim))

        # long term memory
        # if self.percolated_buffer.can_sample(self.config.percolate_len):
        #   percolate = []
        #   for i in range(1):
        #     percolate.append(self.replay_buffer.sample_stack(self.config.percolate_len))
        #   # tensor of shape (batch_size, self.memory_len, num_features)
        #   percolate = np.stack(percolate)
        # else:

        percolate = np.zeros((1, self.config.percolate_len, self.memory_dim))

        ops = [self.sampled_action, self.pretrain_loss]
        if pretraining and np.sum(memory):
          ops.append(self.pretrain_op)

        memory[:,:,-1] = np.array(range(memory.shape[-2])) * 0.1
        res = self.sess.run(ops, feed_dict={
          self.observation_placeholder: states[-1][None],
          self.masked_feature_placeholder: masked_feature,
          self.memory_placeholder: memory,
          self.percolate_placeholder: percolate
        })
        action, loss = res[0][0], res[1]
        losses.append(loss)

        next_state, reward, done, info = env.step(action)

        if self.config.use_mask:
          next_state = mask_obs(self.config.env_name, next_state)
        masked_feature = next_state[2]

        if self.config.use_critic:
          q_targets = []
          for k in range(self.config.q_K):
            q_targets.append(self.sess.run(self.q_targets[k], feed_dict={
              self.observation_placeholder : next_state[None],
              self.action_placeholder: action[None],
              self.memory_placeholder: memory,
              self.percolate_placeholder: percolate
            }))
          q_target = np.mean(np.array(q_targets)[:,:,action])
          ops = [self.td_error, self.train_critic_op]
          if pretraining:
            ops.append(self.pretrain_op)
          res = self.sess.run(
            ops, feed_dict={
            self.q_target_placeholder : reward + self.config.gamma * q_target,
            self.observation_placeholder : state[None],
            self.action_placeholder: action[None],
            self.memory_placeholder: memory,
            self.percolate_placeholder: percolate,
            self.masked_feature_placeholder: masked_feature
          })
          td_error = res[0]
          ops = [self.train_op]
          if pretraining:
            ops.append(self.pretrain_op)
          self.sess.run(
            ops, feed_dict={
            self.observation_placeholder : state[None],
            self.action_placeholder : action[None],
            self.advantage_placeholder : td_error.squeeze(),
            self.memory_placeholder: memory,
            self.percolate_placeholder: percolate,
            self.masked_feature_placeholder: masked_feature
          })
          if self.config.q_K > 1:
            self.sess.run(self.update_target_op, feed_dict={
              self.q_k_placeholder: t % self.config.q_K
            })

        # for milestone
        if self.config.use_attention:
          idx = self.replay_buffer.store_frame(np.array(list(state) + [0]))
          self.replay_buffer.store_effect(idx, action, reward, done)

        state = next_state
        actions.append(action)
        rewards.append(reward)
        memories.append(memory)
        percolates.append(percolate)

        episode_reward += reward
        t += 1
        if (done or step == self.config.max_ep_len-1):
          episode_rewards.append(episode_reward)
          # if self.config.use_attention:
            # self.percolated_buffer.store_embedding(Q_mem[0], np.prod(state.shape))
          break
        if (not num_episodes) and t == self.config.batch_size:
          break
        if t % (self.config.max_ep_len / 2) == 0: # only update percolated once in awhile
          # if self.config.use_attention:
            # self.percolated_buffer.store_embedding(Q_mem[0], np.prod(state.shape))
          if self.config.use_critic and self.config.q_K == 1:
            self.sess.run(self.update_target_op, feed_dict={
            self.q_k_placeholder: 0
          })

      path = {"observation" : np.array(states),
              "reward" : np.array(rewards),
              "action" : np.array(actions),
              "memory" : np.vstack(memories),
              "percolate": np.vstack(percolates),
      }
      path['loss'] = np.mean(losses)
      paths.append(path)
      episode += 1
      if num_episodes and episode >= num_episodes:
        break

    return paths, episode_rewards


  def get_returns(self, paths):
    """
    Calculate the returns G_t for each timestep

    Args:
            paths: recorded sample paths.  See sample_path() for details.

    Return:
            returns: return G_t for each timestep

    After acting in the environment, we record the observations, actions, and
    rewards. To get the advantages that we need for the policy update, we have
    to convert the rewards into returns, G_t, which are themselves an estimate
    of Q^π (s_t, a_t):

       G_t = r_t + γ r_{t+1} + γ^2 r_{t+2} + ... + γ^{T-t} r_T

    where T is the last timestep of the episode.

    TODO: compute and return G_t for each timestep. Use self.config.gamma.
    """
    all_returns = []
    for path in paths:
      rewards = path["reward"]
      returns = []
      for t in range(len(rewards)):
        G_t = sum((self.config.gamma ** i) * r_t for i, r_t in enumerate(rewards[t:]))
        returns.append(G_t)
      all_returns.append(returns)
    returns = np.concatenate(all_returns)
    return returns

  def calculate_advantage(self, returns, observations):
    """
    Calculate the advantage

    Args:
            returns: all discounted future returns for each step
            observations: observations
    Returns:
            adv: Advantage

    Calculate the advantages, using baseline adjustment if necessary,
    and normalizing the advantages if necessary.
    If neither of these options are True, just return returns.

    TODO:
    If config.use_baseline = False and config.normalize_advantage = False,
    then the "advantage" is just going to be the returns (and not actually
    an advantage).

    if config.use_baseline, then we need to evaluate the baseline and subtract
      it from the returns to get the advantage.
      HINT: evaluate the self.baseline with self.sess.run(...)

    if config.normalize_advantage:
      after doing the above, normalize the advantages so that they have a mean of 0
      and standard deviation of 1.
    """
    adv = returns
    if self.config.use_baseline:
      fd = {
        self.observation_placeholder: observations
      }
      baseline = self.sess.run(self.baseline, feed_dict=fd)
      adv -= baseline
    if self.config.normalize_advantage:
      adv = (adv - np.mean(adv)) / np.std(adv)
    return adv

  def update_baseline(self, returns, observations):
    """
    Update the baseline from given returns and observation.

    Args:
            returns: Returns from get_returns
            observations: observations
    TODO:
      apply the baseline update op with the observations and the returns.
      HINT: Run self.update_baseline_op with self.sess.run(...)
    """
    fd = {
      self.baseline_target_placeholder: returns,
      self.observation_placeholder: observations
    }
    self.sess.run(self.update_baseline_op, feed_dict=fd)

  def train(self):
    """
    Performs training

    You do not have to change or use anything here, but take a look
    to see how all the code you've written fits together!
    """

    last_eval = 0
    last_record = 0
    scores_eval = []

    self.init_averages()
    scores_eval = [] # list of scores computed at iteration time
    avg_loss = 0.0
    with tqdm(total=self.config.num_batches) as t:
      for i in range(self.config.num_batches):
        # collect a minibatch of samples
        paths, total_rewards = self.sample_path(self.env, pretraining=True)
        scores_eval = scores_eval + total_rewards
        if not self.config.use_critic:
          observations = np.concatenate([path["observation"] for path in paths])
          actions = np.concatenate([path["action"] for path in paths])
          rewards = np.concatenate([path["reward"] for path in paths])
          memories = np.concatenate([path["memory"] for path in paths])
          percolates = np.concatenate([path["percolate"] for path in paths])

          # compute Q-val estimates (discounted future returns) for each time step
          returns = self.get_returns(paths)
          advantages = self.calculate_advantage(returns, observations)

          # run training operations
          if self.config.use_baseline:
            self.update_baseline(returns, observations)

          self.sess.run(self.train_op, feed_dict={
                        self.observation_placeholder : observations,
                        self.action_placeholder : actions,
                        self.advantage_placeholder : advantages,
                        self.memory_placeholder: memories,
                        self.percolate_placeholder: percolates})

        # tf stuff
        if (i % self.config.summary_freq == 0):
          self.update_averages(total_rewards, scores_eval)
          self.record_summary(i)

        # compute reward statistics for this batch and log
        avg_reward = np.mean(total_rewards)
        sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
        loss = np.mean([path['loss'] for path in paths])
        avg_loss = ((avg_loss * i) + loss) / (i + 1)
        t.set_postfix(loss='{:05.3f}'.format(float(avg_loss)), reward='{:04.2f} +/- {:04.2f}'.format(avg_reward, sigma_reward))
        t.update()
        # msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
        # self.logger.info(msg)

        if  self.config.record and (last_record > self.config.record_freq):
          self.logger.info("Recording...")
          last_record = 0
          self.record()

    self.logger.info("- Training done.")
    export_plot(scores_eval, "Score", config.env_name, self.config.plot_output)

  def evaluate(self, env=None, num_episodes=1):
    """
    Evaluates the return for num_episodes episodes.
    Not used right now, all evaluation statistics are computed during training
    episodes.
    """
    if env == None:
      env = self.env
    paths, rewards = self.sample_path(env, num_episodes)
    avg_reward = np.mean(rewards)
    sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
    msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
    self.logger.info(msg)
    return avg_reward

  def record(self):
     """
     Recreate an env and record a video for one episode
     """
     env = gym.make(self.config.env_name)
     env = gym.wrappers.Monitor(env, self.config.record_path, video_callable=lambda x: True, resume=True)
     self.evaluate(env, 1)

  def run(self):
    """
    Apply procedures of training for a PG.
    """
    # initialize
    self.initialize()
    # record one game at the beginning
    if self.config.record:
      self.record()
    # model
    self.train()
    # record one game at the end
    if self.config.record:
      self.record()

if __name__ == '__main__':
    args = parser.parse_args()

    assert not (args.use_baseline and args.use_critic), \
      "baseline and critic are mutually exclusive"

    config = get_config(
      args.env_name,
      use_baseline=args.use_baseline,
      use_critic=args.use_critic,
      use_attention=args.use_attention,
      use_mask=args.use_mask
    )
    env = gym.make(config.env_name)
    # train model
    model = PG(env, config)
    model.run()
