"""GPU2-only Flex mapper patch backed by persistent one-GPU actors."""

from srcGPU5_7.ray_fitness import persistent_actor_mapper


def enable_persistent_gpu_fitness(
    regressor_class,
    *,
    actors,
    exact_cache,
    pre_batch_size,
    cheap_batch_size,
    exact_batch_size,
    exact_wave_first_limit,
    exact_wave_second_limit,
    full_exact_enabled,
):
    """Replace Flex's transient task mapper only in the GPU5 driver process."""

    def register_map(self, toolbox):
        if self.multiprocessing:
            toolbox.register(
                "map",
                persistent_actor_mapper,
                actors=actors,
                exact_cache=exact_cache,
                pre_batch_size=int(pre_batch_size),
                cheap_batch_size=int(cheap_batch_size),
                exact_batch_size=int(exact_batch_size),
                exact_wave_first_limit=int(exact_wave_first_limit),
                exact_wave_second_limit=int(exact_wave_second_limit),
                full_exact_enabled=bool(full_exact_enabled),
            )
        else:
            raise RuntimeError("GPU5 persistent actors require multiprocessing=True")

    regressor_class._GPSymbolicRegressor__register_map = register_map
