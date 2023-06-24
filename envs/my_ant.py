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
        factor_dist = 1.0,
        factor_healthy = 0.2,
        factor_ctrl = 0.15,
        factor_time = 0.5,
        exp_dist = -1,
        exp_ctrl = -1.5,
        exp_time = -0.75,
        reward_dist = 0,
        reward_time = 0,
        reward_total = 0,
        action = 0,
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

        self.action = action

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

        self.do_simulation(action, self.frame_skip)

        self.action = action

        vec = self.get_body_com("torso")[:2] - self.get_body_com("target")[:2] # Distancia entre el torso y el objetivo
        reward_dist = math.exp(self.exp_dist * (np.linalg.norm(vec) )) * self.factor_dist # Calculo de la recompensa en base a la distancia
        #healthy_reward = self.healthy_reward * self.factor_healthy # Calculo de la recompensa healthy

        #ctrl_cost = math.exp(self.exp_ctrl * self.control_cost(action)) * self.factor_ctrl  # Calculo del castigo por control
        ctrl_cost = (1 - math.exp(-0.3 * self.control_cost(action))) * self.factor_ctrl

        rewards = reward_dist       # Suma de las recompensas
        costs = ctrl_cost           # Suma de los castigos
        reward = rewards - costs    # Calculo de la recompensa final

        """
        print("Action:          ", 0.5 * np.sum(np.square(action)))
        print("ctrl_cost:       ", ctrl_cost)
        print("Real ctrl_cost:  ", (1 - math.exp(-0.3 * self.control_cost(action))) * self.factor_ctrl)
        print("************************************+")
        """
        
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

        observation = self._get_obs()

        info = {
            "reward_forward": reward_dist,
            "reward_ctrl": -ctrl_cost,
            #"reward_contact": -contact_cost,
            #"reward_survive": healthy_reward,
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        observations = np.concatenate((position[:-2], velocity[:-2], self.action, position[-2:], self.get_body_com("torso")[:2] - self.get_body_com("target")[:2]))

        return observations

    def reset_model(self):
        self.timesteps = 0

        self.reward_dist = 0
        self.reward_time = 0
        self.reward_total = 0

        ran = random.uniform( 0, 2 * math.pi ) # Numero aleatorio entre 0 y 2*pi (radianes de una circuferencia)

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