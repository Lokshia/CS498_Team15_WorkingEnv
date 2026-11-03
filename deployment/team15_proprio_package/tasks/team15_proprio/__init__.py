"""Register the independent proprioceptive holding simulation task."""
from .contract import TASK_NAME, enforce_simulation_cli
enforce_simulation_cli()
from booster_deploy.utils.registry import register_task
from .task import ProprioHoldControllerCfg
register_task(TASK_NAME, ProprioHoldControllerCfg())
