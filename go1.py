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
        factor_healthy = 0.2,
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
    def contact_forces(self):
        raw_contact_forces = self.sim.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces)
        )
        return contact_cost
    
    def lista_contactos(self):
        num = 0

        for X in range(self.sim.data.ncon):
            i = self.sim.data.contact[X]
            print("Elem " + str(num) + "-> " + str(i.geom1) + ":" + str(i.geom2))
            num = num + 1
            
    
    
    def contact_count(self):
        #self.lista_contactos()

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

        contacto_trunk   = False

        contacto_FR_thigh   = False
        contacto_FL_thigh   = False
        contacto_RR_thigh   = False
        contacto_RL_thigh   = False
        self.num_contact = self.sim.data.ncon

        for X in range(self.sim.data.ncon):
            i = self.sim.data.contact[X]
            if (i.geom1 == id_floor and i.geom2 == 5) or (i.geom2 == id_floor and i.geom1 == 5):
                contacto_trunk      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 6) or (i.geom2 == id_floor and i.geom1 == 6):
                contacto_trunk      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 7) or (i.geom2 == id_floor and i.geom1 == 7):
                contacto_trunk      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 8) or (i.geom2 == id_floor and i.geom1 == 8):
                contacto_trunk      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 9) or (i.geom2 == id_floor and i.geom1 == 9):
                contacto_trunk      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 10) or (i.geom2 == id_floor and i.geom1 == 10):
                contacto_trunk      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 11) or (i.geom2 == id_floor and i.geom1 == 11):
                contacto_trunk      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 14) or (i.geom2 == id_floor and i.geom1 == 14):
                contacto_trunk      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 15) or (i.geom2 == id_floor and i.geom1 == 15):
                contacto_trunk      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 16) or (i.geom2 == id_floor and i.geom1 == 16):
                contacto_trunk      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 17) or (i.geom2 == id_floor and i.geom1 == 17):
                contacto_trunk      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 18) or (i.geom2 == id_floor and i.geom1 == 18):
                contacto_trunk      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 19) or (i.geom2 == id_floor and i.geom1 == 19):
                contacto_trunk      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 20) or (i.geom2 == id_floor and i.geom1 == 20):
                contacto_trunk      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 23) or (i.geom2 == id_floor and i.geom1 == 23):
                contacto_trunk      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 24) or (i.geom2 == id_floor and i.geom1 == 24):
                contacto_trunk      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 25) or (i.geom2 == id_floor and i.geom1 == 25):
                contacto_trunk      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 26) or (i.geom2 == id_floor and i.geom1 == 26):
                contacto_trunk      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 27) or (i.geom2 == id_floor and i.geom1 == 27):
                contacto_trunk      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 28) or (i.geom2 == id_floor and i.geom1 == 28):
                contacto_trunk      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 29) or (i.geom2 == id_floor and i.geom1 == 29):
                contacto_trunk      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 30) or (i.geom2 == id_floor and i.geom1 == 30):
                contacto_trunk      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 33) or (i.geom2 == id_floor and i.geom1 == 33):
                contacto_trunk      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 34) or (i.geom2 == id_floor and i.geom1 == 34):
                contacto_trunk      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 35) or (i.geom2 == id_floor and i.geom1 == 35):
                contacto_trunk      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 36) or (i.geom2 == id_floor and i.geom1 == 36):
                contacto_trunk      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 37) or (i.geom2 == id_floor and i.geom1 == 37):
                contacto_trunk      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 38) or (i.geom2 == id_floor and i.geom1 == 38):
                contacto_trunk      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 39) or (i.geom2 == id_floor and i.geom1 == 39):
                contacto_trunk      = True
                break
            if (i.geom1 == id_floor and i.geom2 == 40) or (i.geom2 == id_floor and i.geom1 == 40):
                contacto_trunk      = True
                break
            if (i.geom1 == id_floor and i.geom2 == id_trunk) or (i.geom2 == id_floor and i.geom1 == id_trunk):
                contacto_trunk      = True
                break
            if (i.geom1 == id_floor and i.geom2 == id_FR_thigh) or (i.geom2 == id_floor and i.geom1 == id_FR_thigh):
                contacto_FR_thigh   = True
                break
            if (i.geom1 == id_floor and i.geom2 == id_FL_thigh) or (i.geom2 == id_floor and i.geom1 == id_FL_thigh):
                contacto_FL_thigh   = True
                break
            if (i.geom1 == id_floor and i.geom2 == id_RR_thigh) or (i.geom2 == id_floor and i.geom1 == id_RR_thigh):
                contacto_RR_thigh   = True
                break
            if (i.geom1 == id_floor and i.geom2 == id_RL_thigh) or (i.geom2 == id_floor and i.geom1 == id_RL_thigh):
                contacto_RL_thigh   = True
                break

        unhealthy_contact = contacto_trunk or contacto_FR_thigh or contacto_FL_thigh or contacto_RR_thigh or contacto_RL_thigh

        """
        print("*****************")
        print("id_floor:    ", id_floor)
        print("id_trunk:    ", id_trunk)
        print("id_FR_thigh: ", id_FR_thigh)
        print("id_FL_thigh: ", id_FL_thigh)
        print("id_RR_thigh: ", id_RR_thigh)
        print("id_RL_thigh: ", id_RL_thigh)
        print("id_FR_calf:  ", id_FR_calf)
        print("id_FL_calf:  ", id_FL_calf)
        print("id_RR_calf:  ", id_RR_calf)
        print("id_RL_calf:  ", id_RL_calf)
        print("----------------------")
        print("trunk:       ", contacto_trunk)
        print("FR_thigh:    ", contacto_FR_thigh)
        print("FL_thigh:    ", contacto_FL_thigh)
        print("RR_thigh:    ", contacto_RR_thigh)
        print("RL_thigh:    ", contacto_RL_thigh)
        print("Contador:    ", self.num_contact)
        """

        return unhealthy_contact
    
    @property
    def is_healthy(self):
        unhealthy_contact = self.contact_count()

        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z) and not unhealthy_contact
        return is_healthy
    
    @property
    def target_reach(self):
        dist = np.linalg.norm(self.get_body_com("trunk")[:2] - self.get_body_com("target")[:2])
        target_reach = dist < self.delta
        return target_reach

    @property
    def done(self):
        done = not self.is_healthy or self.target_reach if self._terminate_when_unhealthy else False
        return done
    
    def step(self, action):
        self.timesteps = self.timesteps + 1

        self.do_simulation(action, self.frame_skip)

        vec = self.get_body_com("trunk")[:2] - self.get_body_com("target")[:2] # Distancia entre el torso y el objetivo
        reward_dist = math.exp(self.exp_dist * (np.linalg.norm(vec) - self.delta )) * self.factor_dist # Calculo de la recompensa en base a la distancia
        healthy_reward = self.healthy_reward * self.factor_healthy # Calculo de la recompensa healthy

        ctrl_cost = math.exp(self.exp_ctrl * self.control_cost(action)) * self.factor_ctrl  # Calculo del castigo por control
        #contact_cost = self.contact_cost        # Castigo por contacto que sera 0
        contact_cost = math.exp(-3 * (self.num_contact**-2.25)) * 0.15 if self.num_contact != 0 else 0
        
        time_cost = (self.timesteps >= 400) * (1 - math.exp( self.exp_time * ( (self.timesteps - 400 )/100) )) * self.factor_time

        rewards = reward_dist + healthy_reward              # Suma de las recompensas
        costs = ctrl_cost + contact_cost + time_cost        # Suma de los castigos
        reward = rewards - costs                            # Calculo de la recompensa final

        done = self.done

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
        contact_force = self.contact_forces.flat.copy()

        ncom = self.num_contact = self.sim.data.ncon
        
        #observations = np.concatenate((position[:-2], velocity[:-2], contact_force, position[-2:], self.get_body_com("trunk")[:2] - self.get_body_com("target")[:2], np.array([self.timesteps])))
        observations = np.concatenate((position[:-2], velocity[:-2], np.array([ncom]), position[-2:], self.get_body_com("trunk")[:2] - self.get_body_com("target")[:2], np.array([self.timesteps])))

        return observations

    def reset_model(self):
        self.timesteps = 0

        self.reward_dist = 0
        self.reward_time = 0
        self.reward_total = 0

        ran = random.uniform( 0, 2 * math.pi )

        self.target_x = 4 * math.cos(ran)
        self.target_y = 4 * math.sin(ran)
        
        goal = np.array([self.target_x, self.target_y])

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        """
        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv
        )
        """

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