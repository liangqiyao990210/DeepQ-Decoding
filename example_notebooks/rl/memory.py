from __future__ import absolute_import
from collections import deque, namedtuple
import warnings
import random

import numpy as np

# This is to be understood as a transition: Given `state0`, performing `action`
# yields `reward` and results in `state1`, which might be `terminal`.
Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1')


def sample_batch_indexes(low, high, size, priorities=None, alpha=0.0):
    """Return a sample of (size) unique elements between low and high

        # Argument
            low (int): The minimum value for our samples
            high (int): The maximum value for our samples
            size (int): The number of samples to pick
            priorities (list of floats): Priorities of sampled items to use. 
                None if using uniform choice.

        # Returns
            A list of samples of length size, with values between low and high
        """
    if high - low >= size:
        # We have enough data. Draw without replacement, that is each index is unique in the
        # batch. We cannot use `np.random.choice` here because it is horribly inefficient as
        # the memory grows. See https://github.com/numpy/numpy/issues/2764 for a discussion.
        # `random.sample` does the same thing (drawing without replacement) and is way faster.
        r = range(low, high)
        if priorities is None:
            batch_idxs = random.sample(r, size)
        else:
            # Discard the first priority, since the starting state0 corresponding to it no longer is in the buffer.
            # assert high - low == len(priorities), "Length of priorities must match the size of the index sampling window."
            probabilities = np.power(np.array(priorities), alpha)[2:]
            probabilities = probabilities / np.sum(probabilities)
            batch_idxs = np.random.choice(r, size, p=probabilities)
    else:
        # Not enough data. Help ourselves with sampling from the range, but the same index
        # can occur multiple times. This is not good and should be avoided by picking a
        # large enough warm-up phase. Since this only occurs at the beginning of training,
        # we ignore priorities (the priorities should be all the same at this point).
        warnings.warn('Not enough entries to sample without replacement. Consider increasing your warm-up phase to avoid oversampling!')
        batch_idxs = np.random.random_integers(low, high - 1, size=size)
    assert len(batch_idxs) == size
    return batch_idxs


class RingBuffer(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(maxlen)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Return element of buffer at specific index

        # Argument
            idx (int): Index wanted

        # Returns
            The element of buffer at given index
        """
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def __setitem__(self, idx, v):
        """Set element of buffer at specific index to a new value v

        # Argument
            idx (int): Index of element to set
            v (int): New value of item
        """
        if idx < 0 or idx >= self.length:
            raise KeyError()
        self.data[(self.start + idx) % self.maxlen] = v

    # This could potentially be a performance bottleneck if since it is called every time
    # memory.append() is called.
    def get_max(self):
        """Return maximum value of buffer.

        # Returns
            The maximal value contained in the buffer.
        """
        if self.length > 0:
            return max([v for v in self.data if v is not None])
        else:
            return 1

    def append(self, v):
        """Append an element to the buffer

        # Argument
            v (object): Element to append
        """
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


def zeroed_observation(observation):
    """Return an array of zeros with same shape as given observation

    # Argument
        observation (list): List of observation
    
    # Return
        A np.ndarray of zeros with observation.shape
    """
    if hasattr(observation, 'shape'):
        return np.zeros(observation.shape)
    elif hasattr(observation, '__iter__'):
        out = []
        for x in observation:
            out.append(zeroed_observation(x))
        return out
    else:
        return 0.


class Memory(object):
    def __init__(self, window_length, ignore_episode_boundaries=False):
        self.window_length = window_length
        self.ignore_episode_boundaries = ignore_episode_boundaries

        self.recent_observations = deque(maxlen=window_length)
        self.recent_terminals = deque(maxlen=window_length)

    def sample(self, batch_size, batch_idxs=None):
        raise NotImplementedError()

    def append(self, observation, action, reward, terminal, training=True):
        self.recent_observations.append(observation)
        self.recent_terminals.append(terminal)

    def get_recent_state(self, current_observation):
        """Return list of last observations

        # Argument
            current_observation (object): Last observation

        # Returns
            A list of the last observations
        """
        # This code is slightly complicated by the fact that subsequent observations might be
        # from different episodes. We ensure that an experience never spans multiple episodes.
        # This is probably not that important in practice but it seems cleaner.
        state = [current_observation]
        idx = len(self.recent_observations) - 1
        for offset in range(0, self.window_length - 1):
            current_idx = idx - offset
            current_terminal = self.recent_terminals[current_idx - 1] if current_idx - 1 >= 0 else False
            if current_idx < 0 or (not self.ignore_episode_boundaries and current_terminal):
                # The previously handled observation was terminal, don't add the current one.
                # Otherwise we would leak into a different episode.
                break
            state.insert(0, self.recent_observations[current_idx])
        while len(state) < self.window_length:
            state.insert(0, zeroed_observation(state[0]))
        return state

    def get_config(self):
        """Return configuration (window_length, ignore_episode_boundaries) for Memory
        
        # Return
            A dict with keys window_length and ignore_episode_boundaries
        """
        config = {
            'window_length': self.window_length,
            'ignore_episode_boundaries': self.ignore_episode_boundaries,
        }
        return config

class SequentialMemory(Memory):
    def __init__(self, limit, enable_prioritized_replay=False, **kwargs):
        super(SequentialMemory, self).__init__(**kwargs)
        
        self.limit = limit
        self.enable_prioritized_replay = enable_prioritized_replay

        # Do not use deque to implement the memory. This data structure may seem convenient but
        # it is way too slow on random access. Instead, we use our own ring buffer implementation.
        self.actions = RingBuffer(limit)
        self.rewards = RingBuffer(limit)
        self.terminals = RingBuffer(limit)
        self.observations = RingBuffer(limit)
        if self.enable_prioritized_replay:
            self.priorities = RingBuffer(limit)

    def sample(self, batch_size, batch_idxs=None, alpha=0.0):
        """Return a randomized batch of experiences

        # Argument
            batch_size (int): Size of the all batch
            batch_idxs (int): Indexes to extract
        # Returns
            A list of experiences randomly selected
        """
        # It is not possible to tell whether the first state in the memory is terminal, because it
        # would require access to the "terminal" flag associated to the previous state. As a result
        # we will never return this first state (only using `self.terminals[0]` to know whether the
        # second state is terminal).
        # In addition we need enough entries to fill the desired window length.
        assert self.nb_entries >= self.window_length + 2, 'not enough entries in the memory'

        if batch_idxs is None:
            # Draw random indexes such that we have enough entries before each index to fill the
            # desired window length.
            if self.enable_prioritized_replay:
                # Get priorities starting from current "idx" of the RingBuffer
                priorities = [self.priorities[idx] for idx in range(self.nb_entries)]
            
                batch_idxs = sample_batch_indexes(self.window_length, self.nb_entries - 1, size=batch_size, priorities=priorities, alpha=alpha)
            else:
                batch_idxs = sample_batch_indexes(
                        self.window_length, self.nb_entries - 1, size=batch_size)
        batch_idxs = np.array(batch_idxs) + 1
        assert np.min(batch_idxs) >= self.window_length + 1
        assert np.max(batch_idxs) < self.nb_entries
        assert len(batch_idxs) == batch_size

        # Create experiences
        sampled_indexes = []
        experiences = []
        for idx in batch_idxs:
            terminal0 = self.terminals[idx - 2]
            while terminal0:
                # Skip this transition because the environment was reset here. Select a new, random
                # transition and use this instead. This may cause the batch to contain the same
                # transition twice.
                
                if self.enable_prioritized_replay:
                    idx = sample_batch_indexes(self.window_length + 1, self.nb_entries, size=1, priorities=priorities, alpha=alpha)[0]
                else:
                    idx = sample_batch_indexes(self.window_length + 1, self.nb_entries, size=1)[0]
                terminal0 = self.terminals[idx - 2]
            assert self.window_length + 1 <= idx < self.nb_entries

            # This code is slightly complicated by the fact that subsequent observations might be
            # from different episodes. We ensure that an experience never spans multiple episodes.
            # This is probably not that important in practice but it seems cleaner.
            state0 = [self.observations[idx - 1]]
            for offset in range(0, self.window_length - 1):
                current_idx = idx - 2 - offset
                assert current_idx >= 1
                current_terminal = self.terminals[current_idx - 1]
                if current_terminal and not self.ignore_episode_boundaries:
                    # The previously handled observation was terminal, don't add the current one.
                    # Otherwise we would leak into a different episode.
                    break
                state0.insert(0, self.observations[current_idx])
            while len(state0) < self.window_length:
                state0.insert(0, zeroed_observation(state0[0]))
            action = self.actions[idx - 1]
            reward = self.rewards[idx - 1]
            terminal1 = self.terminals[idx - 1]

            # Okay, now we need to create the follow-up state. This is state0 shifted on timestep
            # to the right. Again, we need to be careful to not include an observation from the next
            # episode if the last state is terminal.
            state1 = [np.copy(x) for x in state0[1:]]
            state1.append(self.observations[idx])

            assert len(state0) == self.window_length
            assert len(state1) == len(state0)
            sampled_indexes.append(idx)
            experiences.append(Experience(state0=state0, action=action, reward=reward,
                                          state1=state1, terminal1=terminal1))
        assert len(experiences) == batch_size
        return sampled_indexes, experiences

    def append(self, observation, action, reward, terminal, training=True):
        """Append an observation to the memory

        # Argument
            observation (dict): Observation returned by environment
            action (int): Action taken to obtain this observation
            reward (float): Reward obtained by taking this action
            terminal (boolean): Is the state terminal
        """ 
        super(SequentialMemory, self).append(observation, action, reward, terminal, training=training)
        
        # This needs to be understood as follows: in `observation`, take `action`, obtain `reward`
        # and weather the next state is `terminal` or not.
        if training:
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(terminal)
            if self.enable_prioritized_replay:
                self.priorities.append(self.priorities.get_max())

    @property
    def nb_entries(self):
        """Return number of observations

        # Returns
            Number of observations
        """
        return len(self.observations)

    def get_config(self):
        """Return configurations of SequentialMemory

        # Returns
            Dict of config
        """
        config = super(SequentialMemory, self).get_config()
        config['limit'] = self.limit
        return config

    def update_priorities(self, indices, td_errors, eps=1e-3):
        """Update priorities of items in the buffer.
       
        Parameters:
            indices (list of indices): indices of data that was sampled from the buffer to be updated
            td_errors (list of floats): td_errors for the data points that were sampled
            eps (float): small number to prevent priority staying as 0
        """
        new_priorities = np.abs(td_errors) + eps

        # Update priorities
        for idx_to_update, new_priority in zip(indices, new_priorities):
            self.priorities[idx_to_update] = new_priority


class EpisodeParameterMemory(Memory):
    def __init__(self, limit, **kwargs):
        super(EpisodeParameterMemory, self).__init__(**kwargs)
        self.limit = limit

        self.params = RingBuffer(limit)
        self.intermediate_rewards = []
        self.total_rewards = RingBuffer(limit)

    def sample(self, batch_size, batch_idxs=None):
        """Return a randomized batch of params and rewards

        # Argument
            batch_size (int): Size of the all batch
            batch_idxs (int): Indexes to extract
        # Returns
            A list of params randomly selected and a list of associated rewards
        """
        if batch_idxs is None:
            batch_idxs = sample_batch_indexes(0, self.nb_entries, size=batch_size)
        assert len(batch_idxs) == batch_size

        batch_params = []
        batch_total_rewards = []
        for idx in batch_idxs:
            batch_params.append(self.params[idx])
            batch_total_rewards.append(self.total_rewards[idx])
        return batch_params, batch_total_rewards

    def append(self, observation, action, reward, terminal, training=True):
        """Append a reward to the memory

        # Argument
            observation (dict): Observation returned by environment
            action (int): Action taken to obtain this observation
            reward (float): Reward obtained by taking this action
            terminal (boolean): Is the state terminal
        """
        super(EpisodeParameterMemory, self).append(observation, action, reward, terminal, training=training)
        if training:
            self.intermediate_rewards.append(reward)

    def finalize_episode(self, params):
        """Append an observation to the memory

        # Argument
            observation (dict): Observation returned by environment
            action (int): Action taken to obtain this observation
            reward (float): Reward obtained by taking this action
            terminal (boolean): Is the state terminal
        """
        total_reward = sum(self.intermediate_rewards)
        self.total_rewards.append(total_reward)
        self.params.append(params)
        self.intermediate_rewards = []

    @property
    def nb_entries(self):
        """Return number of episode rewards

        # Returns
            Number of episode rewards
        """
        return len(self.total_rewards)

    def get_config(self):
        """Return configurations of SequentialMemory

        # Returns
            Dict of config
        """
        config = super(SequentialMemory, self).get_config()
        config['limit'] = self.limit
        return config
