# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union, Any, Tuple

import numpy as np
import torch
from gym.wrappers import StepAPICompatibility, TimeLimit

from ..abc import EnvBase, AutoResetEnvBase

from distractor_dmc2gym import make
import gym


class ActionWrapper(gym.ActionWrapper):
    def action(self, action):
        return action.detach().numpy()

    def reverse_action(self, action):
        return torch.from_numpy(action)


class PyTorchWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        return torch.from_numpy(np.ascontiguousarray(observation))


class RealStepsWrapper(gym.Wrapper):
    def __init__(self, env, action_repeat):
        super().__init__(env)
        self._action_repeat = action_repeat

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['actual_env_steps_taken'] = self._action_repeat
        return obs, info

    def step(self, action):
        next_observation, reward, terminated, truncated, step_info = self.env.step(action)
        step_info['actual_env_steps_taken'] = self._action_repeat
        return next_observation, reward, terminated, truncated, step_info


class CustomMethodsWrapper(EnvBase, gym.Wrapper):
    def sample_random_action(self, size=(), np_rng=None) -> Union[float, torch.Tensor]:
        return torch.from_numpy(self.env.action_space.sample())

    def reset(self) -> Tuple[torch.Tensor, 'EnvBase.Info']:
        return self.env.reset()

    def step(self, action) -> Tuple[torch.Tensor, Any, Any, 'EnvBase.Info']:
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def get_random_state(self) -> Any:
        return self.env.np_random

    def set_random_state(self, random_state):
        pass

    def seed(self, seed: Union[int, np.random.SeedSequence]):
        self.env.seed(seed)

    def __init__(self, env, max_episode_length, observation_output, action_repeat):
        super(CustomMethodsWrapper, self).__init__(env)
        self._max_episode_length = max_episode_length
        self._observation_output = observation_output
        self._action_repeat = action_repeat

    @property
    def max_episode_length(self) -> int:
        return self._max_episode_length

    @property
    def observation_output_kind(self) -> 'EnvBase.ObsOutputKind':
        return self._observation_output

    @property
    def action_repeat(self) -> int:
        return self._action_repeat


VARIANTS = [
    'dots_background',
    'dots_foreground',
]


def make_env(spec: str, observation_output_kind: EnvBase.ObsOutputKind, seed,
             max_episode_length, action_repeat, batch_shape):
    # avoid circular imports
    from ..utils import make_batched_auto_reset_env

    for variant in VARIANTS:
        if spec.endswith('_' + variant):
            break
    else:
        # if not break
        raise ValueError(f"Unexpected environment: {spec}")

    domain_name, task_name = spec[:-(len(variant) + 1)].split('_', maxsplit=1)

    kwargs = dict(
        domain_name=domain_name,
        task_name=task_name,
        frame_skip=action_repeat,
        height=64,
        width=64,
        camera_id=0,
        from_pixels=True,
        environment_kwargs=None,
        visualize_reward=False,
        channels_first=True
    )

    if variant == 'dots_background':
        kwargs.update(
            distraction_source="dots",
            distraction_location="background",
            difficulty="hard"
        )
    elif variant == "dots_foreground":
        kwargs.update(
            distraction_source="dots",
            distraction_location="foreground",
            difficulty="hard"
        )
    else:
        raise ValueError(f"Unexpected environment: {spec}")

    def create_env(seed):
        env = make(**kwargs)
        env = TimeLimit(env, max_episode_steps=max_episode_length)
        env = RealStepsWrapper(env, action_repeat=action_repeat)
        env = PyTorchWrapper(env)
        env = ActionWrapper(env)
        env = StepAPICompatibility(env, output_truncation_bool=False)
        env = CustomMethodsWrapper(env, action_repeat=action_repeat, observation_output=EnvBase.ObsOutputKind.image_uint8, max_episode_length=max_episode_length)
        return env

    return make_batched_auto_reset_env(
        create_env, seed, batch_shape)


__all__ = ['make_env']
