from importlib import import_module


class LazyTrainConfig:
    def __init__(self, module_name: str):
        self.module_name = module_name

    def __call__(self):
        module = import_module(self.module_name)
        return module.TrainConfig()


CONFIG_MAPPING = {
    "ram_insertion": LazyTrainConfig("experiments.ram_insertion.config"),
    "usb_pickup_insertion": LazyTrainConfig("experiments.usb_pickup_insertion.config"),
    "object_handover": LazyTrainConfig("experiments.object_handover.config"),
    "egg_flip": LazyTrainConfig("experiments.egg_flip.config"),
    "pick_cube_sim": LazyTrainConfig("experiments.pick_cube_sim.config"),
}
