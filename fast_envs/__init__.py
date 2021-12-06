from gym.envs.registration import register 

register(id='fast-ant-v0',entry_point='fast_envs.envs:FastAntEnv',) 
register(id='fast-ant-friction-v0',entry_point='fast_envs.envs:FastAntFrictionEnv',) 
register(id='fast-half-cheetah-v0',entry_point='fast_envs.envs:FastHalfCheetahEnv',) 
register(id='fast-humanoid-standup-v0',entry_point='fast_envs.envs:FastHumanoidStandupEnv',) 
register(id='mass-spring-damper-v0',entry_point='fast_envs.envs:MassSpringDamperEnv',) 
register(id='hopping-mass-spring-damper-v0',entry_point='fast_envs.envs:HoppingMassSpringDamperEnv',) 
