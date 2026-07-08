from isaacsim.simulation_app import SimulationApp
import time

print("1 before app")
app = SimulationApp({"headless": False})
print("2 after app")

for i in range(100):
    app.update()

print("3 before close")
app.close()
print("4 after close")