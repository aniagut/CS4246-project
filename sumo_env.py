import traci
import gym
from gym import spaces
import numpy as np
import traci.exceptions

class SumoEnv(gym.Env):
    def __init__(self, config_file, max_steps=500, use_gui=False):
        super(SumoEnv, self).__init__()
        self.config_file = config_file
        self.max_steps = max_steps
        self.current_step = 0
        self.use_gui = use_gui

        # Choose SUMO mode based on the `use_gui` parameter
        if self.use_gui:
            self.sumo_cmd = ["sumo-gui", "-c", self.config_file]
        else:
            self.sumo_cmd = ["sumo", "-c", self.config_file]


        # Define action and observation spaces
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32)

    def reset(self):
        # Close TraCI connection if it's already connected
        try:
            traci.close()
        except traci.exceptions.FatalTraCIError:
            # No connection exists; nothing to close
            pass

        # Start a new TraCI connection
        traci.start(self.sumo_cmd)

        self.current_step = 0
        return self._get_observation()

    def step(self, action):
        try:
            # Set the traffic light phase directly based on the action
            traci.trafficlight.setPhase("TL", int(action))

            # Advance simulation
            traci.simulationStep()
            self.current_step += 1

            # Get observation
            obs = self._get_observation()

            # Calculate reward and total waiting time
            reward, total_waiting_time = self._calculate_reward()

            # Check if the simulation is done
            done = self.current_step >= self.max_steps or traci.simulation.getMinExpectedNumber() == 0

            # Return the total waiting time in the info dictionary
            return obs, reward, done, {'total_waiting_time': total_waiting_time}
        except traci.exceptions.FatalTraCIError as e:
            print(f"TraCI Error: {e}")
            self.close()
            raise e

    def _get_observation(self):
        # Collect positions and speeds of up to 10 vehicles as observation
        vehicle_ids = traci.vehicle.getIDList()[:10]  # Limiting to 10 vehicles
        obs = []
        for vehicle_id in vehicle_ids:
            pos = traci.vehicle.getPosition(vehicle_id)
            speed = traci.vehicle.getSpeed(vehicle_id)
            obs.extend([pos[0], pos[1], speed])

        # Pad observation to have exactly 30 elements if fewer vehicles are present
        obs = np.pad(obs, (0, 30 - len(obs)), mode='constant')
        return np.array(obs, dtype=np.float32)

    def _calculate_reward(self):
        # Calculate total waiting time
        total_waiting_time = sum(traci.vehicle.getWaitingTime(veh_id) for veh_id in traci.vehicle.getIDList())
        waiting_penalty = -total_waiting_time  # Negative for reward function

        # Detect teleportations
        teleport_count = traci.simulation.getStartingTeleportNumber()
        teleport_penalty = -10 * teleport_count

        reward = waiting_penalty + teleport_penalty

        print(f"Step {self.current_step}: Waiting Penalty = {waiting_penalty}, Teleportations = {teleport_count}")

        return reward, total_waiting_time

    def close(self):
        try:
            traci.close()
        except traci.exceptions.FatalTraCIError:
            pass
