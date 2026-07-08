import argparse
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run without GUI")
args = parser.parse_args()

print("Starting Isaac Sim launch test...")

try:
    # Try newer explicit import path first
    try:
        from isaacsim.simulation_app import SimulationApp
        print("Imported SimulationApp from isaacsim.simulation_app")
    except Exception:
        from isaacsim import SimulationApp
        print("Imported SimulationApp from isaacsim")

    simulation_app = SimulationApp({"headless": args.headless})
    print("PASS: Isaac Sim launched successfully.")

    # Do a tiny update loop so we know the app is alive
    for i in range(10):
        simulation_app.update()
        print(f"Step {i + 1}/10 OK")

    simulation_app.close()
    print("PASS: Isaac Sim closed cleanly.")

except Exception as e:
    print("FAIL: Isaac Sim launch test failed.")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    traceback.print_exc()
    raise