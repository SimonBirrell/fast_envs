import pathlib

from gym.envs.mujoco import AntEnv
from gym import utils
from gym.envs.mujoco import mujoco_env              
    
class FastAntFrictionEnv(AntEnv):

    def __init__(self):
        path = pathlib.Path(__file__).parent.absolute()
        mujoco_env.MujocoEnv.__init__(self, str(path) + "/assets/fast-ant-friction.xml", 30)
        utils.EzPickle.__init__(self)


