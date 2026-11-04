"""Register the native, no-vision holding policy in booster_deploy."""
from booster_deploy.utils.registry import register_task
from .contract import TASK_NAME
from .task import Team15HoldControllerCfg
register_task(TASK_NAME, Team15HoldControllerCfg())
