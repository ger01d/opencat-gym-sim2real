from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from opencat_gym_env_sim2real import OpenCatGymEnv

if __name__ == '__main__':
    # Create OpenCatGym environment from class
    parallel_env = 1
    env = make_vec_env(OpenCatGymEnv, n_envs=parallel_env)
    model = PPO.load("trained_agent_PPO")

    obs = env.reset()

    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
