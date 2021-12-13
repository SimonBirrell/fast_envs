from gym.envs.registration import register 

register(id='fast-ant-v0',entry_point='fast_envs.envs:FastAntEnv',) 
register(id='fast-ant-friction-v0',entry_point='fast_envs.envs:FastAntFrictionEnv',) 
register(id='fast-half-cheetah-v0',entry_point='fast_envs.envs:FastHalfCheetahEnv',) 
register(id='fast-humanoid-standup-v0',entry_point='fast_envs.envs:FastHumanoidStandupEnv',) 
register(id='mass-spring-damper-v0',entry_point='fast_envs.envs:MassSpringDamperEnv',) 
register(id='hopping-mass-spring-damper-v0',entry_point='fast_envs.envs:HoppingMassSpringDamperEnv',) 
register(id='hopping-mass-spring-damper-pertubations-v0',entry_point='fast_envs.envs:HoppingMassSpringDamperPertubationsEnv',) 
register(id='three-masses-two-springs-v0',entry_point='fast_envs.envs:ThreeMassTwoSpringsEnv',) 
register(id='two-masses-one-spring-v0',entry_point='fast_envs.envs:TwoMassOneSpringsEnv',) 
register(id='multipendulum-v0',entry_point='fast_envs.envs:MultipendulumEnv',) 
