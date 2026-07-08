import argparse
import traceback

print("Starting Isaac Lab validation...")

try:
    from isaaclab.app import AppLauncher
    print("PASS: imported AppLauncher from isaaclab.app")
except Exception as exc:
    print("FAIL: could not import Isaac Lab AppLauncher")
    print(type(exc).__name__, exc)
    traceback.print_exc()
    raise SystemExit(1)


parser = argparse.ArgumentParser(description="Minimal Isaac Lab validation script.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

try:
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    print("PASS: Isaac Lab launched Isaac Sim app")
except Exception as exc:
    print("FAIL: AppLauncher could not launch Isaac Sim")
    print(type(exc).__name__, exc)
    traceback.print_exc()
    raise SystemExit(1)


try:
    import torch
    import isaaclab
    import isaaclab_assets

    print("PASS: imported torch")
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    print("PASS: imported isaaclab")
    print("PASS: imported isaaclab_assets")

    for i in range(100):
        simulation_app.update()

    print("PASS: Isaac Lab update loop ran 100 frames")

except Exception as exc:
    print("FAIL: Isaac Lab runtime check failed")
    print(type(exc).__name__, exc)
    traceback.print_exc()
    raise SystemExit(1)

finally:
    simulation_app.close()
    print("Isaac Lab app closed")