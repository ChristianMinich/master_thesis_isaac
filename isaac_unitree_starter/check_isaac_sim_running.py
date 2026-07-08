from isaacsim.simulation_app import SimulationApp
import time

app = SimulationApp({
    "headless": False,
    "width": 1280,
    "height": 720,
})

print("Isaac Sim opened. Keeping it alive...")

try:
    while app.is_running():
        app.update()
        time.sleep(0.01)

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    app.close()
    print("Isaac Sim closed.")