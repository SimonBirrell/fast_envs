import pathlib

from gym.envs.mujoco import HumanoidStandupEnv
from gym import utils
from gym.envs.mujoco import mujoco_env              
    
class FastHumanoidStandupEnv(HumanoidStandupEnv):

    def __init__(self):
        path = pathlib.Path(__file__).parent.absolute()
        mujoco_env.MujocoEnv.__init__(self, str(path) + "/assets/fast-humanoidstandup.xml", 30)
        utils.EzPickle.__init__(self)


