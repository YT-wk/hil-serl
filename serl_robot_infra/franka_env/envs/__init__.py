def _missing_realsense_class(class_name):
    class MissingRealSenseDependency:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "pyrealsense2 is required for real Franka environments. "
                "Install pyrealsense2 or use a simulation-only environment."
            )

    MissingRealSenseDependency.__name__ = class_name
    return MissingRealSenseDependency


try:
    from franka_env.envs.franka_env import FrankaEnv, DefaultEnvConfig
except ModuleNotFoundError as exc:
    if exc.name != "pyrealsense2":
        raise
    FrankaEnv = _missing_realsense_class("FrankaEnv")
    DefaultEnvConfig = _missing_realsense_class("DefaultEnvConfig")

try:
    from franka_env.envs.franka_wrench_env import FrankaWrenchEnv
except ModuleNotFoundError as exc:
    if exc.name != "pyrealsense2":
        raise
    FrankaWrenchEnv = _missing_realsense_class("FrankaWrenchEnv")
