"""V8 Flex mapper patch backed by persistent actors (V7's gpu_ray reduced to the three stages)."""

from src4DCartPoleV10.ray_fitness import persistent_actor_mapper


def enable_persistent_gpu_fitness(
    regressor_class,
    *,
    actors,
    pre_batch_size,
    det4_batch_size,
    cpu_actors=None,
):
    """Replace Flex's transient task mapper only in the driver process."""

    def register_map(self, toolbox):
        if self.multiprocessing:
            toolbox.register(
                "map",
                persistent_actor_mapper,
                actors=actors,
                pre_batch_size=int(pre_batch_size),
                det4_batch_size=int(det4_batch_size),
                cpu_actors=cpu_actors,
            )
        else:
            raise RuntimeError("V8 persistent actors require multiprocessing=True")

    regressor_class._GPSymbolicRegressor__register_map = register_map
