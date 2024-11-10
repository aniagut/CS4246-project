from stable_baselines3 import DQN
from sumo_env import SumoEnv
import traci


def run_pretrained_dqn(model_path="DQN_MODEL3", config_file="intersection.sumocfg", max_steps=500):
    # Load the pre-trained model
    model = DQN.load(model_path)

    # Initialize the SUMO environment
    env = SumoEnv(config_file=config_file, max_steps=max_steps, use_gui=True)

    # Reset the environment to start a new simulation
    obs = env.reset()

    # Variables to track performance metrics
    total_reward = 0
    total_waiting_time = 0
    total_queue_length = 0
    throughput = 0

    # Run the simulation using the pre-trained model
    done = False
    while not done:
        # Predict the action using the loaded model
        action, _states = model.predict(obs, deterministic=True)

        # Take a step in the environment
        obs, reward, done, info = env.step(action)

        # Accumulate metrics
        total_reward += reward

        total_waiting_time += sum(traci.vehicle.getWaitingTime(veh_id) for veh_id in traci.vehicle.getIDList())
        total_queue_length += sum(traci.lane.getLastStepHaltingNumber(lane) for lane in traci.lane.getIDList())
        throughput += len(traci.simulation.getArrivedIDList())

    # Calculate average waiting time and queue length
    avg_waiting_time = total_waiting_time / max_steps
    avg_queue_length = total_queue_length / max_steps

    # Print the final metrics
    print("\nSimulation Results with Pre-trained DQN Model:")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Avg Waiting Time: {avg_waiting_time:.2f}")
    print(f"Avg Queue Length: {avg_queue_length:.2f}")
    print(f"Total Throughput: {throughput}")

    # Close the environment
    env.close()


# Run the simulation using the pre-trained model
try:
    run_pretrained_dqn()
except Exception as e:
    print(f"An error occurred during simulation: {e}")
finally:
    traci.close()
