import numpy as np
import spinup.algos.tf1.ppo.core as core
from spinup.utils.mpi_tools import mpi_statistics_scalar


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_dim = [i for i in obs_dim]
        self.act_dim = [i for i in act_dim]
        self.size = size
        self.obs_buf = np.zeros([size] + self.obs_dim, dtype=np.float32)
        self.act_buf = np.zeros([size] + self.act_dim, dtype=np.float32)
        self.adv_buf = np.zeros((size), dtype=np.float32)
        self.rew_buf = np.zeros((size), dtype=np.float32)
        self.ret_buf = np.zeros((size), dtype=np.float32)
        self.val_buf = np.zeros((size), dtype=np.float32)
        self.logp_buf = np.zeros((size), dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # bufcore.combined_shape(obs_dim)fer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf,
                self.ret_buf, self.logp_buf]


class FPOEpochBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99):
        self.obs_dim = [i for i in obs_dim]
        self.act_dim = [i for i in act_dim]
        self.size = size
        self.obs_buf = np.zeros([size+1] + [2] + self.obs_dim, dtype=np.float32)
        self.act_buf = np.zeros([size] + [2] + self.act_dim, dtype=np.float32)
        self.rew_buf = np.zeros((size, 2), dtype=np.float32)
        self.ret_buf = np.zeros((size, 2), dtype=np.float32)
        self.logp_buf = np.zeros((size, 2), dtype=np.float32)
        self.done_buf = np.zeros((size, 2), dtype = np.float32)
        self.gamma = gamma
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, logp, done=0):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.logp_buf[self.ptr] = logp
        self.done_buf[self.ptr] = done
        self.ptr += 1

    def finish_store(self, obs):
        self.obs_buf[-1] = obs

    def finish_path(self, last_val=[0, 0]):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], [last_val], axis=0)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # bufcore.combined_shape(obs_dim)fer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        return [self.obs_buf, self.act_buf, self.rew_buf,
                self.logp_buf, self.done_buf, self.ret_buf]


class FPOHistoryBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, local_step, epochs, gamma=0.99, lam=0.95, replay_epochs=None):
        self.obs_dim = [i for i in obs_dim]
        self.act_dim = [i for i in act_dim]
        self.local_step = local_step
        self.epochs = epochs + 1
        if replay_epochs is None:
            self.replay_epochs = self.epochs
        else:
            self.replay_epochs = replay_epochs
        self.obs_buf = np.zeros([self.epochs] + [self.local_step + 1] + [2] + self.obs_dim, dtype=np.float32)
        self.act_buf = np.zeros([self.epochs] + [self.local_step] + [2] + self.act_dim, dtype=np.float32)
        self.rew_buf = np.zeros((self.epochs, self.local_step, 2), dtype=np.float32)
        self.logp_buf = np.zeros((self.epochs, self.local_step, 2), dtype=np.float32)
        self.done_buf = np.zeros((self.epochs, self.local_step, 2), dtype=np.float32)
        self.adv_buf = np.zeros((self.epochs, self.local_step, 2), dtype=np.float32)
        self.ret_buf = np.zeros((self.epochs, self.local_step, 2), dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr = 0

    def store(self, obs, act, rew, logp, done, ret):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.logp_buf[self.ptr] = logp
        self.done_buf[self.ptr] = done
        self.ret_buf[self.ptr] = ret
        self.ptr += 1
        self.ptr = self.ptr % self.epochs


    def finish_path(self, vals, log_pis):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        start_round = max(0, self.ptr-self.replay_epochs)
        end_round = self.ptr
        path_slice = slice(start_round, end_round)

        rews = self.rew_buf[path_slice]
        vals = np.reshape(vals, [end_round-start_round, self.local_step + 1, 2])
        done = self.done_buf[path_slice]

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews + (1 - done) * self.gamma * vals[:, 1:] - vals[:, :-1]
        for i in reversed(range(deltas.shape[1]-1)):
            deltas[:, i] = self.lam * self.gamma * (1-done[:, i]) * deltas[:, i+1] + deltas[:, i]

        adv_placeholder = deltas
        self.adv_buf[path_slice] = adv_placeholder
        self.logp_buf[path_slice] = np.reshape(log_pis, [end_round-start_round, self.local_step, 2])

    def pre_get(self):
        start_round = max(0, self.ptr - self.replay_epochs)
        end_round = self.ptr
        path_slice = slice(start_round, end_round)
        print(path_slice)
        return [np.reshape(self.obs_buf[path_slice, :-1], [-1] + self.obs_dim),
                np.reshape(self.obs_buf[path_slice], [-1] + self.obs_dim),
                np.reshape(self.act_buf[path_slice], [-1])]

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        start_round = max(0, self.ptr - self.replay_epochs)
        end_round = self.ptr
        path_slice = slice(start_round, end_round)
        print(path_slice)

        adv_buf = np.reshape(self.adv_buf[path_slice], [-1])
        adv_mean, adv_std = mpi_statistics_scalar(adv_buf)
        adv_buf = (adv_buf - adv_mean)/adv_std
        return [np.reshape(self.obs_buf[path_slice, :-1], [-1] + self.obs_dim),
                np.reshape(self.act_buf[path_slice], [-1] + self.act_dim),
                adv_buf,
                np.reshape(self.ret_buf[path_slice], [-1]),
                np.reshape(self.logp_buf[path_slice], [-1])]


class NFSPHistoryBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, local_step, epochs, replay_epochs=None):
        self.obs_dim = [i for i in obs_dim]
        self.act_dim = [i for i in act_dim]
        self.local_step = local_step
        self.epochs = epochs + 1
        if replay_epochs is None:
            self.replay_epochs = self.epochs
        else:
            self.replay_epochs = replay_epochs
        size = self.epochs * self.local_step
        self.obs_buf = np.zeros([size] + self.obs_dim, dtype=np.float32)
        self.act_buf = np.zeros([size] + self.act_dim, dtype=np.float32)
        self.ptr = 0

    def store(self, obs, act):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.ptr += 1
        self.ptr = self.ptr % (self.epochs*self.local_step)

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        start_ = max(0, self.ptr - self.replay_epochs*self.local_step)
        end_ = self.ptr
        path_slice = slice(start_, end_)
        print(path_slice)

        return [self.obs_buf[path_slice],
                self.act_buf[path_slice]]
