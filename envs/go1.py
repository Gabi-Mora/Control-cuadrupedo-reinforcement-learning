import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

import time

import math
import random

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class GoOneEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        xml_file="go1.xml",
        ctrl_cost_weight=0.5,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.15, 0.5),
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
        factor_healthy = 0.4,
        factor_ctrl = 0.15,
        factor_time = 0.5,
        exp_dist = -1,
        exp_ctrl = -1.5,
        exp_time = -0.75,
        reward_dist = 0,
        reward_time = 0,
        reward_total = 0,
        num_contact = 0,
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

        self.num_contact = num_contact

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    # Función de la recompensa "healthy"
    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    # Coste del control producido
    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.sim.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    # Función para calcular el coste por contacto --> No funciona, solo necesitamos contarlos
    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces)
        )
        return contact_cost
    
    # Función para la detección de los contactos
    def contact_count(self):
        # Identificación de los ID de las diferentes partes del cuerpo del robot
        id_floor    = self.sim.model.geom_name2id("floor")

        id_trunk    = self.sim.model.geom_name2id("trunk")

        id_FR_thigh = self.sim.model.geom_name2id("FR_thigh")
        id_FL_thigh = self.sim.model.geom_name2id("FL_thigh")
        id_RR_thigh = self.sim.model.geom_name2id("RR_thigh")
        id_RL_thigh = self.sim.model.geom_name2id("RL_thigh")
        
        id_FR_calf = self.sim.model.geom_name2id("FR_calf")
        id_FL_calf = self.sim.model.geom_name2id("FL_calf")
        id_RR_calf = self.sim.model.geom_name2id("RR_calf")
        id_RL_calf = self.sim.model.geom_name2id("RL_calf")

        unhealthy_contact = False

        self.num_contact = self.sim.data.ncon # Guardamos el numero de contactos que se ha producido

        # Atravesamos el array de todos los cantactos existentes para determinar si hay alguno que es considerado "unhealthy" (No he encontrado la manera de hacer esto sin tener que hacer esta monstruosidad)
        for X in range(self.sim.data.ncon):
            i = self.sim.data.contact[X]
            if (i.geom1 == id_floor and i.geom2 == 5) or (i.geom2 == id_floor and i.geom1 == 5):
                unhealthy_contact      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 6) or (i.geom2 == id_floor and i.geom1 == 6):
                unhealthy_contact      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 7) or (i.geom2 == id_floor and i.geom1 == 7):
                unhealthy_contact      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 8) or (i.geom2 == id_floor and i.geom1 == 8):
                unhealthy_contact      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 9) or (i.geom2 == id_floor and i.geom1 == 9):
                unhealthy_contact      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 10) or (i.geom2 == id_floor and i.geom1 == 10):
                unhealthy_contact      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 11) or (i.geom2 == id_floor and i.geom1 == 11):
                unhealthy_contact      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 14) or (i.geom2 == id_floor and i.geom1 == 14):
                unhealthy_contact      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 15) or (i.geom2 == id_floor and i.geom1 == 15):
                unhealthy_contact      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 16) or (i.geom2 == id_floor and i.geom1 == 16):
                unhealthy_contact      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 17) or (i.geom2 == id_floor and i.geom1 == 17):
                unhealthy_contact      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 18) or (i.geom2 == id_floor and i.geom1 == 18):
                unhealthy_contact      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 19) or (i.geom2 == id_floor and i.geom1 == 19):
                unhealthy_contact      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 20) or (i.geom2 == id_floor and i.geom1 == 20):
                unhealthy_contact      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 23) or (i.geom2 == id_floor and i.geom1 == 23):
                unhealthy_contact      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 24) or (i.geom2 == id_floor and i.geom1 == 24):
                unhealthy_contact      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 25) or (i.geom2 == id_floor and i.geom1 == 25):
                unhealthy_contact      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 26) or (i.geom2 == id_floor and i.geom1 == 26):
                unhealthy_contact      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 27) or (i.geom2 == id_floor and i.geom1 == 27):
                unhealthy_contact      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 28) or (i.geom2 == id_floor and i.geom1 == 28):
                unhealthy_contact      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 29) or (i.geom2 == id_floor and i.geom1 == 29):
                unhealthy_contact      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 30) or (i.geom2 == id_floor and i.geom1 == 30):
                unhealthy_contact      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 33) or (i.geom2 == id_floor and i.geom1 == 33):
                unhealthy_contact      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 34) or (i.geom2 == id_floor and i.geom1 == 34):
                unhealthy_contact      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 35) or (i.geom2 == id_floor and i.geom1 == 35):
                unhealthy_contact      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 36) or (i.geom2 == id_floor and i.geom1 == 36):
                unhealthy_contact      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 37) or (i.geom2 == id_floor and i.geom1 == 37):
                unhealthy_contact      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 38) or (i.geom2 == id_floor and i.geom1 == 38):
                unhealthy_contact      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 39) or (i.geom2 == id_floor and i.geom1 == 39):
                unhealthy_contact      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 40) or (i.geom2 == id_floor and i.geom1 == 40):
                unhealthy_contact      = True
                break
            if (i.geom1 == id_floor and i.geom2 == id_trunk) or (i.geom2 == id_floor and i.geom1 == id_trunk):
                unhealthy_contact      = True
                break
            if (i.geom1 == id_floor and i.geom2 == id_FR_thigh) or (i.geom2 == id_floor and i.geom1 == id_FR_thigh):
                unhealthy_contact   = True
                break
            if (i.geom1 == id_floor and i.geom2 == id_FL_thigh) or (i.geom2 == id_floor and i.geom1 == id_FL_thigh):
                unhealthy_contact   = True
                break
            if (i.geom1 == id_floor and i.geom2 == id_RR_thigh) or (i.geom2 == id_floor and i.geom1 == id_RR_thigh):
                unhealthy_contact   = True
                break
            if (i.geom1 == id_floor and i.geom2 == id_RL_thigh) or (i.geom2 == id_floor and i.geom1 == id_RL_thigh):
                unhealthy_contact   = True
                break

        return unhealthy_contact
    
    # Función para determinar si el robot esta sano o no. Tiene en cuenta que no hay ningun contacto no deseado y que la altura del robot esta dentro de los limites establecidos
    @property
    def is_healthy(self):
        unhealthy_contact = self.contact_count()

        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z) and not unhealthy_contact
        return is_healthy
    
    # Función que estima si el robot ha alcanzado el objetivo. Si se encuentra a una distancia menor que un delta establecido, se detemina que el robot ha alcanzado el objetivo.
    @property
    def target_reach(self):
        dist = np.linalg.norm(self.get_body_com("trunk")[:2] - self.get_body_com("target")[:2])
        target_reach = dist < self.delta
        return target_reach

    # Función para determinar si el episodio ya ha terminado
    @property
    def done(self):
        done = not self.is_healthy or self.target_reach if self._terminate_when_unhealthy else False
        return done
    
    # Función que se ejecuta en cada paso
    def step(self, action):
        self.timesteps = self.timesteps + 1

        self.do_simulation(action, self.frame_skip)

        vec = self.get_body_com("trunk")[:2] - self.get_body_com("target")[:2]                          # Distancia entre el torso y el objetivo
        reward_dist = math.exp(self.exp_dist * (np.linalg.norm(vec) - self.delta )) * self.factor_dist  # Calculo de la recompensa en base a la distancia
        healthy_reward = self.healthy_reward * self.factor_healthy                                      # Calculo de la recompensa healthy

        ctrl_cost = math.exp(self.exp_ctrl * self.control_cost(action)) * self.factor_ctrl              # Calculo del castigo por control --> Actualmente tiene nulo impacto
        #ctrl_cost = 1 - math.exp(-0.05 * self.control_cost(action)) * self.factor_ctrl                 # Calculo del castigo por control --> Da problemas
        
        contact_cost = math.exp(-3 * (self.num_contact**-2.25)) * 0.05 if self.num_contact != 0 else 0  # Castigo en base a la cantidad de contactos producidos (con el suelo o con sigo mismo)
        
        time_cost = (self.timesteps >= 400) * (1 - math.exp( self.exp_time * ( (self.timesteps - 400 )/100) )) * self.factor_time # Castigo por tiempo

        rewards = reward_dist + healthy_reward              # Suma de las recompensas
        costs = ctrl_cost + contact_cost + time_cost        # Suma de los castigos
        reward = rewards - costs                            # Calculo de la recompensa final

        done = self.done

        # Estados terminales
        if done:
            if not self.is_healthy:
                reward = reward - 100
                """
                print("------------------------")
                print("---- AGENT  CRASHED ----")
                print("------------------------")
                """
            if self.target_reach:
                reward = reward + 200
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

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        ncom = self.num_contact = self.sim.data.ncon # Numero de contactos producidos
        
        observations = np.concatenate((position[:-2], velocity[:-2], np.array([ncom]), position[-2:], self.get_body_com("trunk")[:2] - self.get_body_com("target")[:2], np.array([self.timesteps])))

        return observations

    def reset_model(self):
        self.timesteps = 0

        self.reward_dist = 0
        self.reward_time = 0
        self.reward_total = 0

        ran = random.uniform( 0, 2 * math.pi )

        # Posicionamiento del objetivo en un punto aleatorio dentro de una circuferencia de radio 4
        self.target_x = 4 * math.cos(ran)
        self.target_y = 4 * math.sin(ran)
        
        goal = np.array([self.target_x, self.target_y])

        qpos = self.init_qpos
        qvel = self.init_qvel

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