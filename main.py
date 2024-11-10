import traci
import traci.constants as tc

# Initialize SUMO Simulation
sumo_cmd = ["sumo-gui", "-c", "intersection.sumocfg"]  # or use "sumo" for non-GUI mode
traci.start(sumo_cmd)

# Simulation parameters
simulation_steps = 500  # Define total simulation time steps

# Metrics initialization
total_waiting_time = 0
throughput = 0
total_queue_length = 0
avg_waiting_time = 0
avg_queue_length = 0
total_reward = 0  # New metric to track total reward


# Function to calculate the reward for each step
def calculate_step_reward():
    vehicle_ids = traci.vehicle.getIDList()

    # Calculate total waiting time for all vehicles
    step_waiting_time = sum(traci.vehicle.getWaitingTime(veh) for veh in vehicle_ids)

    # Detect teleportations
    teleport_count = traci.simulation.getStartingTeleportNumber()
    teleport_penalty = -10 * teleport_count

    # Calculate the reward (negative waiting time + teleport penalty)
    step_reward = -step_waiting_time + teleport_penalty
    return step_reward, step_waiting_time


# Function to update vehicle positions and print their status
def simulate_vehicles():
    global total_waiting_time, throughput, total_queue_length, avg_waiting_time, avg_queue_length, total_reward

    for step in range(simulation_steps):
        traci.simulationStep()  # Move simulation by one step

        # Calculate reward and metrics for each step
        step_reward, step_waiting_time = calculate_step_reward()
        total_reward += step_reward
        total_waiting_time += step_waiting_time

        # Calculate throughput (number of vehicles that have completed their routes)
        throughput += len(traci.simulation.getArrivedIDList())

        # Calculate total queue length
        step_queue_length = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in traci.lane.getIDList())
        total_queue_length += step_queue_length

        # Calculate average metrics
        avg_waiting_time = total_waiting_time / (step + 1) if step + 1 > 0 else 0
        avg_queue_length = total_queue_length / (step + 1) if step + 1 > 0 else 0

        print(
            f"Step {step}: Avg Waiting Time: {avg_waiting_time:.2f}, Throughput: {throughput}, "
            f"Avg Queue Length: {avg_queue_length:.2f}, Total Reward: {total_reward:.2f}"
        )

    # Print final metrics after simulation ends
    print("\nFinal Metrics:")
    print(f"Total Throughput: {throughput}")
    print(f"Avg Waiting Time: {avg_waiting_time:.2f}")
    print(f"Avg Queue Length: {avg_queue_length:.2f}")
    print(f"Total Reward: {total_reward:.2f}")

    traci.close()


# Run the simulation
try:
    simulate_vehicles()
except Exception as e:
    print(f"An error occurred during simulation: {e}")
finally:
    traci.close()
