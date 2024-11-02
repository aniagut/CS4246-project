import traci
import time

try:
    traci.start(["sumo-gui", "-c", "intersection.sumocfg"])
    time.sleep(1)
    traci.close()
except Exception as e:
    print(f"Error during connection test: {e}")
