import numpy as np
import gym


class POMDPWrapper(gym.ObservationWrapper):
    def __init__(self, env_name, pomdp_type='POMDP-RV',
                 flicker_prob=0.2, random_noise_sigma=0.1, random_sensor_missing_prob=0.1):
        """
        :param env_name:
        :param pomdp_type:
            1.  POMDP-RV: remove velocity related observation
            2.  POMDP-FLK: obscure the entire observation with a certain probability at each time step with the
                           probability flicker_prob.
            3.  POMDP-RN: each sensor in an observation is disturbed by a random noise Normal ~ (0, sigma).
            4.  POMDP-RSM: each sensor in an observation will miss with a relatively low probability sensor_miss_prob
            5.  POMDP-RV_and_POMDP-FLK:
            6.  POMDP-RV_and_POMDP-RN:
            7.  POMDP-RV_and_POMDP-RSM:
            8.  POMDP-FLK_and_POMDP-RN:
            9.  POMDP-RN_and_POMDP-RSM
            10. POMDP-RSM_and_POMDP-RN:

        """
        super().__init__(gym.make(env_name))
        self.pomdp_type = pomdp_type
        self.flicker_prob = flicker_prob
        self.random_noise_sigma = random_noise_sigma
        self.random_sensor_missing_prob = random_sensor_missing_prob

        if pomdp_type == 'POMDP-RV':
            # Remove Velocity info, comes with the change in observation space.
            self.remain_obs_idx, self.observation_space = self._remove_velocity(env_name)
        elif pomdp_type == 'POMDP-FLK':
            pass
        elif self.pomdp_type == 'POMDP-RN':
            pass
        elif self.pomdp_type == 'POMDP-RSM':
            pass
        elif self.pomdp_type == 'POMDP-RV_and_POMDP-FLK':
            # Remove Velocity Info, comes with the change in observation space.
            self.remain_obs_idx, self.observation_space = self._remove_velocity(env_name)
        elif self.pomdp_type == 'POMDP-RV_and_POMDP-RN':
            # Remove Velocity Info, comes with the change in observation space.
            self.remain_obs_idx, self.observation_space = self._remove_velocity(env_name)
        elif self.pomdp_type == 'POMDP-RV_and_POMDP-RSM':
            # Remove Velocity Info, comes with the change in observation space.
            self.remain_obs_idx, self.observation_space = self._remove_velocity(env_name)
        elif self.pomdp_type == 'POMDP-FLK_and_POMDP-RN':
            pass
        elif self.pomdp_type == 'POMDP-RN_and_POMDP-RSM':
            pass
        elif self.pomdp_type == 'POMDP-RSM_and_POMDP-RN':
            pass
        else:
            raise ValueError("pomdp_type was not specified!")

    def observation(self, obs):
        # Single source of POMDP
        if self.pomdp_type == 'POMDP-RV':
            return obs.flatten()[self.remain_obs_idx]
        elif self.pomdp_type == 'POMDP-FLK':
            # Note: POMDP-FLK is equivalent to:
            #   POMDP-FLK_and_POMDP-RSM, POMDP-RN_and_POMDP-FLK, POMDP-RSM_and_POMDP-FLK
            if np.random.rand() <= self.flicker_prob:
                return np.zeros(obs.shape)
            else:
                return obs.flatten()
        elif self.pomdp_type == 'POMDP-RN':
            return (obs + np.random.normal(0, self.random_noise_sigma, obs.shape)).flatten()
        elif self.pomdp_type == 'POMDP-RSM':
            obs[np.random.rand(len(obs)) <= self.random_sensor_missing_prob] = 0
            return obs.flatten()
        # Multiple source of POMDP
        elif self.pomdp_type == 'POMDP-RV_and_POMDP-FLK':
            # Note: POMDP-RV_and_POMDP-FLK is equivalent to POMDP-FLK_and_POMDP-RV
            # Remove velocity
            new_obs = obs.flatten()[self.remain_obs_idx]
            # Flickering
            if np.random.rand() <= self.flicker_prob:
                return np.zeros(new_obs.shape)
            else:
                return new_obs
        elif self.pomdp_type == 'POMDP-RV_and_POMDP-RN':
            # Note: POMDP-RV_and_POMDP-RN is equivalent to POMDP-RN_and_POMDP-RV
            # Remove velocity
            new_obs = obs.flatten()[self.remain_obs_idx]
            # Add random noise
            return (new_obs + np.random.normal(0, self.random_noise_sigma, new_obs.shape)).flatten()
        elif self.pomdp_type == 'POMDP-RV_and_POMDP-RSM':
            # Note: POMDP-RV_and_POMDP-RSM is equivalent to POMDP-RSM_and_POMDP-RV
            # Remove velocity
            new_obs = obs.flatten()[self.remain_obs_idx]
            # Random sensor missing
            new_obs[np.random.rand(len(new_obs)) <= self.random_sensor_missing_prob] = 0
            return new_obs
        elif self.pomdp_type == 'POMDP-FLK_and_POMDP-RN':
            # Flickering
            if np.random.rand() <= self.flicker_prob:
                new_obs = np.zeros(obs.shape)
            else:
                new_obs = obs
            # Add random noise
            return (new_obs + np.random.normal(0, self.random_noise_sigma, new_obs.shape)).flatten()
        elif self.pomdp_type == 'POMDP-RN_and_POMDP-RSM':
            # Random noise
            new_obs = (obs + np.random.normal(0, self.random_noise_sigma, obs.shape)).flatten()
            # Random sensor missing
            new_obs[np.random.rand(len(new_obs)) <= self.random_sensor_missing_prob] = 0
            return new_obs
        elif self.pomdp_type == 'POMDP-RSM_and_POMDP-RN':
            # Random sensor missing
            obs[np.random.rand(len(obs)) <= self.random_sensor_missing_prob] = 0
            # Random noise
            return (obs + np.random.normal(0, self.random_noise_sigma, obs.shape)).flatten()
        else:
            raise ValueError("pomdp_type was not in ['POMDP-RV', 'POMDP-FLK', 'POMDP-RN', 'POMDP-RSM']!")

    def _remove_velocity(self, env_name):
        # OpenAIGym
        #  1. Classic Control
        if env_name == "Pendulum-v0":
            remain_obs_idx = np.arange(0, 2)
        elif env_name == "Acrobot-v1":
            remain_obs_idx = list(np.arange(0, 4))
        elif env_name == "MountainCarContinuous-v0":
            remain_obs_idx = list([0])
        #  1. MuJoCo
        elif env_name == "HalfCheetah-v3" or env_name == "HalfCheetah-v2":
            remain_obs_idx = np.arange(0, 8)
        elif env_name == "Ant-v3" or env_name == "Ant-v2":
            remain_obs_idx = list(np.arange(0, 13)) + list(np.arange(27, 111))
        elif env_name == 'Walker2d-v3' or env_name == "Walker2d-v2":
            remain_obs_idx = np.arange(0, 8)
        elif env_name == 'Hopper-v3' or env_name == "Hopper-v2":
            remain_obs_idx = np.arange(0, 5)
        elif env_name == "InvertedPendulum-v2":
            remain_obs_idx = np.arange(0, 2)
        elif env_name == "InvertedDoublePendulum-v2":
            remain_obs_idx = list(np.arange(0, 5)) + list(np.arange(8, 11))
        elif env_name == "Swimmer-v3" or env_name == "Swimmer-v2":
            remain_obs_idx = np.arange(0, 3)
        elif env_name == "Thrower-v2":
            remain_obs_idx = list(np.arange(0, 7)) + list(np.arange(14, 23))
        elif env_name == "Striker-v2":
            remain_obs_idx = list(np.arange(0, 7)) + list(np.arange(14, 23))
        elif env_name == "Pusher-v2":
            remain_obs_idx = list(np.arange(0, 7)) + list(np.arange(14, 23))
        elif env_name == "Reacher-v2":
            remain_obs_idx = list(np.arange(0, 6)) + list(np.arange(8, 11))
        elif env_name == 'Humanoid-v3' or env_name == "Humanoid-v2":
            remain_obs_idx = list(np.arange(0, 22)) + list(np.arange(45, 185)) + list(np.arange(269, 376))
        elif env_name == 'HumanoidStandup-v2':
            remain_obs_idx = list(np.arange(0, 22)) + list(np.arange(45, 185)) + list(np.arange(269, 376))
        # PyBulletEnv:
        #   The following is not implemented:
        #       HumanoidDeepMimicBulletEnv - v1
        #       CartPoleBulletEnv - v1
        #       MinitaurBulletEnv - v0
        #       MinitaurBulletDuckEnv - v0
        #       RacecarBulletEnv - v0
        #       RacecarZedBulletEnv - v0
        #       KukaBulletEnv - v0
        #       KukaCamBulletEnv - v0
        #       PusherBulletEnv - v0
        #       ThrowerBulletEnv - v0
        #       StrikerBulletEnv - v0
        #       HumanoidBulletEnv - v0
        #       HumanoidFlagrunBulletEnv - v0
        #       HumanoidFlagrunHarderBulletEnv - v0
        elif env_name == 'HalfCheetahBulletEnv-v0':
            remain_obs_idx = list(set(np.arange(0, 26)) - set(np.arange(3, 6)))
        elif env_name == 'AntBulletEnv-v0':
            remain_obs_idx = list(set(np.arange(0, 28)) - set(np.arange(3, 6)))
        elif env_name == 'HopperBulletEnv-v0':
            remain_obs_idx = list(set(np.arange(0, 15)) - set(np.arange(3, 6)))
        elif env_name == 'Walker2DBulletEnv-v0':
            remain_obs_idx = list(set(np.arange(0, 22)) - set(np.arange(3, 6)))
        elif env_name == 'InvertedPendulumBulletEnv-v0':
            remain_obs_idx = list(set(np.arange(0, 5)) - set([1, 4]))
        elif env_name == 'InvertedDoublePendulumBulletEnv-v0':
            remain_obs_idx = list(set(np.arange(0, 9)) - set([1, 5, 8]))
        elif env_name == 'InvertedPendulumSwingupBulletEnv-v0':
            pass
        elif env_name == 'ReacherBulletEnv-v0':
            remain_obs_idx = list(set(np.arange(0, 9)) - set([6, 8]))
        # PyBulletGym
        #  1. MuJoCo
        elif env_name == 'HalfCheetahMuJoCoEnv-v0':
            remain_obs_idx = np.arange(0, 8)
        elif env_name == 'AntMuJoCoEnv-v0':
            remain_obs_idx = list(np.arange(0, 13)) + list(np.arange(27, 111))
        elif env_name == 'Walker2DMuJoCoEnv-v0':
            remain_obs_idx = np.arange(0, 8)
        elif env_name == 'HopperMuJoCoEnv-v0':
            remain_obs_idx = np.arange(0, 7)
        elif env_name == 'InvertedPendulumMuJoCoEnv-v0':
            remain_obs_idx = np.arange(0, 3)
        elif env_name == 'InvertedDoublePendulumMuJoCoEnv-v0':
            remain_obs_idx = list(np.arange(0, 5)) + list(np.arange(8, 11))
        #  2. Roboschool
        elif env_name == 'HalfCheetahPyBulletEnv-v0':
            remain_obs_idx = list(set(np.arange(0, 26)) - set(np.arange(3, 6)))
        elif env_name == 'AntPyBulletEnv-v0':
            remain_obs_idx = list(set(np.arange(0, 28)) - set(np.arange(3, 6)))
        elif env_name == 'Walker2DPyBulletEnv-v0':
            remain_obs_idx = list(set(np.arange(0, 22)) - set(np.arange(3, 6)))
        elif env_name == 'HopperPyBulletEnv-v0':
            remain_obs_idx = list(set(np.arange(0, 15)) - set(np.arange(3, 6)))
        elif env_name == 'InvertedPendulumPyBulletEnv-v0':
            remain_obs_idx = list(set(np.arange(0, 5)) - set([1, 4]))
        elif env_name == 'InvertedDoublePendulumPyBulletEnv-v0':
            remain_obs_idx = list(set(np.arange(0, 9)) - set([1, 5, 8]))
        elif env_name == 'ReacherPyBulletEnv-v0':
            remain_obs_idx = list(set(np.arange(0, 9)) - set([6, 8]))
        else:
            raise ValueError('POMDP for {} is not defined!'.format(env_name))

        # Redefine observation_space
        obs_low = np.array([-np.inf for i in range(len(remain_obs_idx))], dtype="float32")
        obs_high = np.array([np.inf for i in range(len(remain_obs_idx))], dtype="float32")
        observation_space = gym.spaces.Box(obs_low, obs_high)
        return remain_obs_idx, observation_space

    def reset(self):
        observation = self.env.reset()
        return self.observation(observation)

if __name__ == '__main__':
    # import pybulletgym
    import pybullet_envs
    import gym
    # POMDP-FLK, POMDP-RV, POMDP-RN
    env = POMDPWrapper("HalfCheetahBulletEnv-v0", 'POMDP-RN')  # Pendulum-v0    AntBulletEnv-v0    HalfCheetahBulletEnv-v0
    obs = env.reset()
    env.step(env.action_space.sample())
    print('action_space: {}'.format(env.action_space))
    print('observation_space: {}'.format(env.observation_space))
    print('obs: {}'.format(obs))
