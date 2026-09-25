"""Run with python -m tests.vbd.benchmark_vbd_rod_blocks; pure Torch, no scene.

Reports synchronized median complete-sweep time. ReverseRodModel is the retained
pre-optimization test oracle. Shared GPU load affects timings.
"""

import statistics
import time

import torch

from genesis.engine.solvers.vbd_rod import RodModel, RodParameters, retract
from tests.vbd.test_vbd_rod_blocks import ReverseRodModel


def main():
    torch.set_num_threads(1)
    print(f"torch={torch.__version__} CPU_threads={torch.get_num_threads()}", flush=True)
    for device in ("cpu", "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            continue
        if device == "cuda":
            print(f"GPU={torch.cuda.get_device_name()}", flush=True)
        for count in (5, 16, 32):
            generator = torch.Generator(device=device).manual_seed(193)
            rest = torch.zeros((count, 3), device=device, dtype=torch.float64)
            rest[:, 2] = torch.arange(count, device=device) * 0.01
            quat = torch.zeros((count - 1, 4), device=device, dtype=rest.dtype)
            quat[:, 0] = 1
            model = RodModel(rest, quat, 0.001, RodParameters(1050, 1300, 1900, 2600, 4000, 900))
            oracle = ReverseRodModel(rest, quat, model.radius, model.parameters)
            model.scale += 0.03 * torch.randn(count, device=device, dtype=rest.dtype, generator=generator)
            model.quat = retract(quat, 0.05 * torch.randn((count - 1, 3), device=device,
                                                        dtype=rest.dtype, generator=generator))
            state = model.get_state()
            x = rest + 0.0001 * torch.randn(rest.shape, device=device, dtype=rest.dtype, generator=generator)
            pinned = torch.zeros(count, device=device, dtype=torch.bool)
            samples = [[], []]
            for repeat in range(4):
                # Alternate order to reduce thermal/shared-GPU scheduling bias.
                for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                    implementation = (oracle, model)[index]
                    implementation.set_state(state)
                    implementation.begin(x, torch.zeros_like(x), 0.003, x.new_tensor([0.0, 0.0, -9.81]))
                    if device == "cuda":
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    result = implementation.sweep(x, pinned)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    duration = time.perf_counter() - start
                    if repeat:
                        samples[index].append(duration)
                    if index == 0:
                        expected = result
                        expected_state = implementation.get_state()
                    else:
                        actual = result
                        actual_state = implementation.get_state()
                torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-11)
                torch.testing.assert_close(actual_state.scale, expected_state.scale, atol=1e-12, rtol=1e-11)
                torch.testing.assert_close(actual_state.quat, expected_state.quat, atol=1e-12, rtol=1e-11)
            reverse, optimized = map(statistics.median, samples)
            print(f"{device} nodes={count} reverse_s={reverse:.6f} optimized_s={optimized:.6f} "
                  f"speedup={reverse / optimized:.3f}", flush=True)


if __name__ == "__main__":
    main()
