import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

import time

import math
import random


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class My_AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        xml_file="new_ant.xml",
        ctrl_cost_weight=0.5,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.3, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=False,
        desired_Y = 0,
        episode_time = time.time(),
        target_x = 0,
        target_y = 0,
        delta = 0.5,
        timesteps = 0,
        factor_dist = 0.8,
        factor_healthy = 0.2,
        factor_ctrl = 0.15,
        factor_time = 0.5,
        exp_dist = -1,
        exp_ctrl = -1.5,
        exp_time = -0.75,
        reward_dist = 0,
        reward_time = 0,
        reward_total = 0,
    ):
        utils.EzPickle.__init__(**locals())

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self.desired_Y = desired_Y

        self.episode_time = episode_time

        self.target_x = target_x

        self.target_y = target_y

        self.delta = delta

        self.timesteps = timesteps

        self.factor_dist = factor_dist
        self.factor_healthy = factor_healthy
        self.factor_ctrl = factor_ctrl
        self.factor_time = factor_time
        self.exp_dist = exp_dist
        self.exp_ctrl = exp_ctrl
        self.exp_time = exp_time

        self.reward_dist = reward_dist
        self.reward_time = reward_time
        self.reward_total = reward_total

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost
    
    @property
    def prueba_contacto(self):
        id_floor = self.sim.model.geom_name2id("floor")
        id_blue = self.sim.model.geom_name2id("right_ankle_geom")
        id_purple = self.sim.model.geom_name2id("left_ankle_geom")
        id_green = self.sim.model.geom_name2id("third_ankle_geom")
        id_brown = self.sim.model.geom_name2id("fourth_ankle_geom")

        contacto_blue   = False
        contacto_purple = False
        contacto_green  = False
        contacto_brown  = False
        contador = 0

        print("Size: ", self.sim.data.ncon)

        for X in range(self.sim.data.ncon):
            i = self.sim.data.contact[X]
            if (i.geom1 == id_floor and i.geom2 == id_blue) or (i.geom2 == id_floor and i.geom1 == id_blue):
                contacto_blue = True
            if (i.geom1 == id_floor and i.geom2 == id_purple) or (i.geom2 == id_floor and i.geom1 == id_purple):
                contacto_purple = True
            if (i.geom1 == id_floor and i.geom2 == id_green) or (i.geom2 == id_floor and i.geom1 == id_green):
                contacto_green = True
            if (i.geom1 == id_floor and i.geom2 == id_brown) or (i.geom2 == id_floor and i.geom1 == id_brown):
                print("Distancia: ", i.dist)
                contacto_brown = True

        """
        for i in self.sim.data.contact:
            if (i.geom1 == id_floor and i.geom2 == id_blue) or (i.geom2 == id_floor and i.geom1 == id_blue):
                contacto_blue = True
            if (i.geom1 == id_floor and i.geom2 == id_purple) or (i.geom2 == id_floor and i.geom1 == id_purple):
                contacto_purple = True
            if (i.geom1 == id_floor and i.geom2 == id_green) or (i.geom2 == id_floor and i.geom1 == id_green):
                contacto_green = True
            if (i.geom1 == id_floor and i.geom2 == id_brown) or (i.geom2 == id_floor and i.geom1 == id_brown):
                print("Distancia: ", i.dist)
                contacto_brown = True
        """

        contador = contacto_blue + contacto_purple + contacto_green + contacto_brown

        print("*****************")
        print("id_floor: ", id_floor)
        print("id_blue: ", id_blue)
        print("id_purple: ", id_purple)
        print("id_green: ", id_green)
        print("id_brown: ", id_brown)
        print("----------------------")
        print("Azul:        ", contacto_blue)
        print("Morado:      ", contacto_purple)
        print("Verde:       ", contacto_green)
        print("Marron:      ", contacto_brown)
        print("Contador:    ", contador)

        return 0

    @property
    def contact_forces(self):
        raw_contact_forces = self.sim.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        #print(len(contact_forces))
        #sensor = self.get_sensor_sensordata()
        #data_all = self.sim.data.contact[0]
        #print(sensor)
        #print("Lo nuevo:")
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces)
        )
        return contact_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        return is_healthy
    
    @property
    def target_reach(self):
        dist = np.linalg.norm(self.get_body_com("torso")[:2] - self.get_body_com("target")[:2])
        target_reach = dist < self.delta
        return target_reach

    @property
    def done(self):
        done = not self.is_healthy or self.target_reach if self._terminate_when_unhealthy else False
        return done
    
    def step(self, action):
        self.timesteps = self.timesteps + 1
        #print("timesteps: ", self.timesteps)

        self.do_simulation(action, self.frame_skip)

        vec = self.get_body_com("torso")[:2] - self.get_body_com("target")[:2] # Distancia entre el torso y el objetivo
        reward_dist = math.exp(self.exp_dist * (np.linalg.norm(vec) - self.delta )) * self.factor_dist # Calculo de la recompensa en base a la distancia
        healthy_reward = self.healthy_reward * self.factor_healthy # Calculo de la recompensa healthy

        ctrl_cost = math.exp(self.exp_ctrl * self.control_cost(action)) * self.factor_ctrl  # Calculo del castigo por control
        contact_cost = self.contact_cost        # Castigo por contacto que sera 0

        #ejemplo = self.prueba_contacto

        #print("ctrl_cost: ", ctrl_cost)

        #actual_time = time.time()
        #time_cost = (1 - math.exp( -0.5 * abs(actual_time - self.episode_time) )) * 0.5
        #time_cost = (1 - math.exp( self.exp_time * (self.timesteps/100) )) * self.factor_time
        time_cost = (self.timesteps >= 400) * (1 - math.exp( self.exp_time * ( (self.timesteps - 400 )/100) )) * self.factor_time

        #print("time: ", abs(actual_time - self.episode_time))

        rewards = reward_dist + healthy_reward  # Suma de las recompensas
        costs = ctrl_cost + contact_cost + time_cost        # Suma de los castigos
        reward = rewards - costs                # Calculo de la recompensa final

        done = self.done
        fall = False
        reached = False

        if done:
            if not self.is_healthy:
                reward = reward - 100
                fall = True
                """
                print("------------------------")
                print("---- AGENT  CRASHED ----")
                print("------------------------")
                """
            if self.target_reach:
                reward = reward + 200
                reached = True
                """
                print("************************")
                print("**** TARGET REACHED ****")
                print("************************")
                """
        else:
            if self.timesteps == 1000:
                reward = reward - 50

        self.reward_dist = self.reward_dist + reward_dist
        self.reward_time = self.reward_time + time_cost
        self.reward_total = self.reward_total + reward

        observation = self._get_obs()

        info = {
            "reward_forward": reward_dist,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_survive": healthy_reward,
        }

        return observation, reward, done, info

    """
    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        forward_reward = x_velocity
        healthy_reward = self.healthy_reward

        # diagonal_cost defined
        # diagonal_cost  = abs(y_velocity) * 5

        # y_error = math.exp( -2 *  ( abs(self.desired_Y - xy_position_after[1]) ) )

        ap1 = ( self.target_x -  xy_position_after[0]) ** 2
        ap2 = ( self.target_y -  xy_position_after[1]) ** 2

        forward_reward = math.sqrt(ap1 + ap2)
        
        print("-----------------------------")
        print("Distancia:              ", forward_reward)

        forward_reward = math.exp( -1 *  forward_reward )

        # rewards = forward_reward + healthy_reward
        # rewards = forward_reward + healthy_reward + y_error
        rewards = forward_reward + healthy_reward
        
        costs = ctrl_cost + contact_cost
        # costs = ctrl_cost + contact_cost + diagonal_cost # Added diagnoal_cost

        reward = rewards - costs

        
        print("Reconpensa Seguimiento: ", forward_reward)
        print("Reconpensa Healthy:     ", healthy_reward)
        print("Posicion Agente X:      ", xy_position_after[0])
        print("Posicion Agente Y:      ", xy_position_after[1])
        print("Posicion Target X:      ", self.target_x)
        print("Posicion Target Y:      ", self.target_y)
        print("Recompensa:             ", reward)
        

        done = self.done
        observation = self._get_obs()
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_survive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        return observation, reward, done, info
    """

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        contact_force = self.contact_forces.flat.copy()

        print(contact_force)

        #if self._exclude_current_positions_from_observation:
            #position = position[3:]

        """self.actual_time = time.time()

        self.target_x = 4 * math.cos(self.actual_time)
        self.target_y = 4 * math.sin(self.actual_time)

        goal = np.array([self.target_x, self.target_y, 0.6])
        print("goal: ", len(goal))
        position[-3:] = goal
        print(goal)

        velocity[-3] = 0
        self.set_state(position, velocity)
        """

        #position = position[:-2]
        #velocity = velocity[:-2]

        #observations = np.concatenate((position, velocity, contact_force))
        #observations = np.concatenate((position, velocity, contact_force, np.array([self.target_x]), np.array([self.target_y])))
        #observations = np.concatenate((position, velocity, contact_force, self.get_body_com("target")[:2], self.get_body_com("torso")[:2] - self.get_body_com("target")[:2]))
        #observations = np.concatenate((position, velocity, contact_force, self.get_body_com("target")[:2], self.get_body_com("torso")[:2] - self.get_body_com("target")[:2], np.array([self.timesteps])))
        observations = np.concatenate((position[:-2], velocity[:-2], contact_force, position[-2:], self.get_body_com("torso")[:2] - self.get_body_com("target")[:2], np.array([self.timesteps])))

        #print("POS:    ", position[-2:])
        #print("TARGET: ", self.get_body_com("target")[:2])

        return observations

    def reset_model(self):
        #self.episode_time = time.time()
        #self.episode_time = self.timesteps # 352!!!!
        self.timesteps = 0

        self.reward_dist = 0
        self.reward_time = 0
        self.reward_total = 0

        ran = random.uniform( 0, 2 * math.pi ) # Numero aleatorio entre 0 y 2*pi (radianes de una circuferencia)
        #ran = random.uniform( math.pi / 2, (3 * math.pi / 2) )

        self.target_x = 4 * math.cos(ran) # Coordenada X en base a un radian aleatorio en una circuferencia radio 4
        self.target_y = 4 * math.sin(ran) # Coordenada Y en base a un radian aleatorio en una circuferencia radio 4
        
        goal = np.array([self.target_x, self.target_y])

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv
        )

        qpos[-2:] = goal
        qvel[-2:] = 0

        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)