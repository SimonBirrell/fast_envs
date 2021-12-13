import gym

from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.classic_control import rendering

import numpy as np

class TwoMassOneSpringsEnv(gym.Env):
	metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }
	def __init__(
		self, 
		n_cycles=20, 
		y0=[0.5, 0.0], 
		y_dot0=[0.0, 0.0], 
		phase0='stance', 
		warnings=True,
		get_pertubations=None
		): 

		# Physics constants
		self.sim_m1 = 0.5 	# mass (kg)
		self.sim_m2 = 0.1 	# mass (kg)
		self.sim_k = 200 	# Spring constant
		self.sim_c = 0.1	# Damping constant (Ns/m)
		self.sim_L0 = 0.5	# Natural length of spring
		self.sim_dt = 0.001 # dt for simulation
		self.sim_g = 9.8 	# N/m2

		# Simulation constants
		self.n_cycles = n_cycles
		self.error_threshold = 10**-4

		# Initial conditions
		self.y0 = np.array(y0)
		self.y_dot0 = np.array(y_dot0)
		self.phase0 = phase0

		# State
		self.sim_y = None
		self.sim_y_dot = None
		self.sim_phase = None

		# Limits
		self.max_force = np.array([2]) 
		self.x_threshold = 4.8
		self.obs_high = np.array([self.x_threshold, np.finfo(np.float32).max])

		# External forces
		self.get_pertubations = get_pertubations
		print("-------------get_pertubations-------")
		print(get_pertubations)

		# Gym stuff
		self.observation_space = spaces.Box(-np.float32(self.obs_high), np.float32(self.obs_high), dtype=np.float32)
		self.action_space = spaces.Box(-np.float32(self.max_force), np.float32(self.max_force), dtype=np.float32)

		# Visualization
		self.viewer = None
		self.screen_width = 600
		self.screen_height = 400
		self.world_height = 1.0
		self.scale = self.screen_height / self.world_height
		self.screen_floor_y = 0.2 * self.scale
		self.spring_x = self.screen_width / 2
		self.warnings = warnings

	def reset(self):
		self.sim_y = self.y0
		self.sim_y_dot = self.y_dot0
		self.sim_phase = self.phase0
		self.pertubations = np.array([0.0, 0.0, 0.0])
		self.sim_t = 0
		return self._get_state()

	def step(self, action):
		f = action[0]
		m1 = self.sim_m1
		m2 = self.sim_m2
		g = self.sim_g
		y = self.sim_y
		k = self.sim_k
		c = self.sim_c
		L0 = self.sim_L0
		y = self.sim_y
		y_dot = self.sim_y_dot
		p = self.pertubations
		dt = self.sim_dt
		info = {'phase': self.sim_phase, 'phase_change': 0}
		if not self.action_space.contains(action):
			if self.warnings:
				print("ILLEGAL ACTION CLIPPED : ", action)
			action = np.clip(action,-1,1)
		assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

		if self.sim_y is None or self.sim_y_dot is None or self.sim_phase is None:
			raise("Must call env.reset() before env.step()")

		p = np.zeros(y.shape) if self.get_pertubations is None else self.get_pertubations(self.sim_t)
		info['p'] = p

		for t in range(self.n_cycles):
			#print("action", action)
			#print("t=",t)
			#print("f",f)
			#print("L10",L10)
			#print("y",y)
			#print("y_dot",y_dot)
			#print("phase", self.sim_phase)
			#print("-(y[0]-y[1]-L10)*k1/m1",-(y[0]-y[1]-L10)*k1/m1)
			#print("-g+f/m1",-g+f/m1)
			#print("-c1*(y_dot[0]-y_dot[1])/m1",-c1*(y_dot[0]-y_dot[1])/m1)
			if self.sim_phase == 'stance':
				# Matlab: [-(y(1)-y(2)-L0)*k/m1-g+f/m1-c*(y(3)-y(4))/m1, 0]
				y_dot_dot = np.array([
					-(y[0]-y[1]-L0)*k/m1-g+f/m1-c*(y_dot[0]-y_dot[1])/m1, 
					0
					])
			elif self.sim_phase == 'flight':
				# Matlab: [-(y(1)-y(2)-L0)*k/m1-g+f/m1-c*(y(3)-y(4))/m1,(y(1)-y(2)-L0)*k/m2-g+c*(y(3)-y(4))/m2]
				y_dot_dot = np.array([
					-(y[0]-y[1]-L0)*k/m1-g+f/m1-c*(y_dot[0]-y_dot[1])/m1,
					(y[0]-y[1]-L0)*k/m2-g+c*(y_dot[0]-y_dot[1])/m2
					])

			# Euler Method
			#print("y_dot", y_dot)
			#print("y_dot_dot", y_dot_dot)
			#print("dt",dt)
			y_dot = y_dot + y_dot_dot * dt
			y = y + y_dot * dt

			if self.sim_phase == 'stance':
				if m2*g-(y[0]-L0)*k<self.error_threshold:
					self.sim_phase = 'flight'
					info['phase_change'] = 1
			elif self.sim_phase == 'flight':
				if y[1]<self.error_threshold:
					self.sim_phase = 'stance'
					y[1] = 0.0
					y_dot[1] = 0.0
					info['phase_change'] = 1

		self.sim_y = y
		self.sim_y_dot = y_dot	
		reward = 0
		done = False
		self.sim_t = self.sim_t + 1
		#print("y",y,"phase",self.sim_phase, "y..", y_dot_dot)
 
		return self._get_state(), reward, done, info

	def _get_state(self):
		return np.concatenate((np.array(self.sim_y), np.array(self.sim_y_dot))).squeeze()

	def _y_to_screen_y(self, y):
		return [self.screen_floor_y, self.screen_floor_y] + y * self.scale	

	def _create_mass(self, r, g, b, radius):	
		rend_mass = rendering.make_circle()
		rend_mass.set_color(r, g, b)
		transform = rendering.Transform()
		rend_mass.add_attr(transform)
		self.viewer.add_geom(rend_mass)
		return transform

	def _create_spring(self, r, g, b):
		rend_spring = rendering.Line((0, 0), (0, -1))
		rend_spring.set_color(0,100,0)
		transform = rendering.Transform()
		rend_spring.add_attr(transform)
		self.viewer.add_geom(rend_spring)	
		return transform

	def _create_floor(self):	
		rend_floor = rendering.Line((0, self.screen_floor_y), (self.screen_width, self.screen_floor_y))
		rend_floor.set_color(0,0,0)
		self.viewer.add_geom(rend_floor)
			
	def render(self, mode='human'):

		if self.viewer is None:
			self.viewer = rendering.Viewer(self.screen_width, self.screen_height)
			self._create_floor()

			self.rend_mass1_transform = self._create_mass(100, 0, 0, 20)
			self.rend_mass2_transform = self._create_mass(50, 50, 0, 20)
			self.rend_spring1_transform = self._create_spring(0, 100, 0)

		spring_tops = self._y_to_screen_y(self.sim_y)
			
		self.rend_mass1_transform.set_translation(self.spring_x, spring_tops[0])
		self.rend_mass2_transform.set_translation(self.spring_x, spring_tops[1])
		self.rend_spring1_transform.set_translation(self.spring_x, spring_tops[0])
		self.rend_spring1_transform.set_scale(1.0, spring_tops[0]-spring_tops[1])

		return self.viewer.render(return_rgb_array = mode =='rgb_array')

	def close(self):
		if self.viewer:
			self.viewer.close()
			self.viewer = None

	def get_obs(self):
		return self._get_state()


