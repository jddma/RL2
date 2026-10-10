"""Transformacion de observaciones Dict a un vector float32 normalizado de siete elementos."""
import numpy as np

ENV_ID = 'hill_racing_env/HillRacing-v0'
ENV_KWARGS = dict(action_space='discrete_3', reward_function='distance', reward_type='aggressive', max_steps=1200)

def preprocess_observation(obs):
    if not isinstance(obs, dict):
        state = np.asarray(obs, dtype=np.float32)
        if state.shape != (7,):
            raise ValueError(f'Se esperaba estado (7,), recibido {state.shape}')
        return state
    p = np.asarray(obs['chassis_position']).reshape(-1)
    a = np.asarray(obs['chassis_angle']).reshape(-1)
    w = np.asarray(obs['wheels_speed']).reshape(-1)
    g = np.asarray(obs['on_ground']).reshape(-1)
    if (len(p), len(a), len(w), len(g)) != (2, 1, 2, 2):
        raise ValueError('Dimensiones de observacion distintas a las documentadas')
    return np.array([p[0]/1000., p[1]/700., a[0]/360., w[0]/(13*np.pi+.1), w[1]/(13*np.pi+.1), float(g[0]), float(g[1])], dtype=np.float32)

class FlatObservationWrapper:
    """Wrapper ligero sin depender de herencia Gymnasium."""
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = None
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return preprocess_observation(obs), info
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return preprocess_observation(obs), reward, terminated, truncated, info
    def close(self):
        return self.env.close()

def make_env():
    import gymnasium as gym
    import hill_racing_env  # noqa: F401 - registra el entorno
    return FlatObservationWrapper(gym.make(ENV_ID, **ENV_KWARGS))
