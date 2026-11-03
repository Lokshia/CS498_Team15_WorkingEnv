"""Run the new actor with one physical free box in official K1 simulation."""
import argparse
import json
import math
import torch

from .task import ProprioHoldControllerCfg
from .simulator import ProprioHoldMujocoController


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mass-kg", type=float, default=.5)
    parser.add_argument("--paused", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.mass_kg) or args.mass_kg < .001:
        parser.error("--mass-kg must be finite and at least 0.001 kg")
    torch.set_num_threads(1)
    controller = ProprioHoldMujocoController(ProprioHoldControllerCfg(), payload_mass_kg=args.mass_kg)
    controller._paused = args.paused
    print(json.dumps(controller.scene_audit, indent=2))
    controller.run()


if __name__ == "__main__":
    main()
