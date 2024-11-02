import traci
import traci.constants as tc

# Initialize SUMO Simulation
sumo_cmd = ["sumo-gui", "-c", "intersection.sumocfg"]  # or use "sumo" for non-GUI mode
traci.start(sumo_cmd)

# Simulation parameters
simulation_steps = 500  # Define total simulation time steps


# Function to update vehicle positions and print their status
def simulate_vehicles():
    for step in range(simulation_steps):
        traci.simulationStep()  # Move simulation by one step

        # Retrieve and print details for each vehicle
        for vehicle_id in traci.vehicle.getIDList():
            position = traci.vehicle.getPosition(vehicle_id)
            speed = traci.vehicle.getSpeed(vehicle_id)
            route = traci.vehicle.getRoute(vehicle_id)
            lane_position = traci.vehicle.getLanePosition(vehicle_id)

            print(
                f"Step {step}: Vehicle {vehicle_id} is at position {position} on route {route} with speed {speed} m/s and lane position {lane_position}")

        # Additional logic: Slow down vehicles approaching intersections
        for vehicle_id in traci.vehicle.getIDList():
            upcoming_traffic_light = traci.vehicle.getNextTLS(vehicle_id)  # Get info about traffic lights ahead
            if upcoming_traffic_light:
                distance_to_light = upcoming_traffic_light[0][2]  # Distance to the next traffic light
                light_state = upcoming_traffic_light[0][3]  # Light state ("r" for red, "g" for green)

                # Slow down if the light is red and within 10 meters of the light
                if light_state == 'r' and distance_to_light < 10:
                    traci.vehicle.slowDown(vehicle_id, 0, 2)  # Reduce speed to 0 within 2 seconds

    traci.close()


# Run the simulation
try:
    simulate_vehicles()
except Exception as e:
    print(f"An error occurred during simulation: {e}")
finally:
    traci.close()
