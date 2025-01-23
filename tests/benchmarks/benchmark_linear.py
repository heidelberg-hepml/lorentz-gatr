import torch, time
import numpy as np
import matplotlib.pyplot as plt

from gatr.primitives.linear import (
    equi_linear,
    linear_v2,
    linear_v2_compiled,
    USE_FULLY_CONNECTED_SUBGROUP,
)
from gatr.layers.linear import NUM_PIN_LINEAR_BASIS_ELEMENTS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def benchmark(n_particles):
    # create data
    batchsize, channels = n_particles, 32
    x = torch.randn(batchsize, channels, 16, device=DEVICE, dtype=DTYPE)
    coeffs = torch.randn(
        channels, channels, NUM_PIN_LINEAR_BASIS_ELEMENTS, device=DEVICE, dtype=DTYPE
    )

    if DEVICE == torch.device("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    out_v1 = equi_linear(x, coeffs, use_v2=False)
    dt1 = time.time() - t0

    if DEVICE == torch.device("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    out_v2 = linear_v2(x, coeffs)
    dt2 = time.time() - t0

    if DEVICE == torch.device("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    out_v2 = linear_v2_compiled(x, coeffs)
    dt3 = time.time() - t0

    # torch.testing.assert_close(out_v1, out_v2)
    return dt1, dt2, dt3


if __name__ == "__main__":
    benchmark(10000)

    results = {}
    n_particles = 2 ** np.arange(15)
    for n in list(n_particles):
        dt1, dt2, dt3 = [], [], []
        for _ in range(50):
            t1, t2, t3 = benchmark(n)
            dt1.append(t1)
            dt2.append(t2)
            dt3.append(t3)

        results[n] = {"v1": dt1, "v2": dt2, "v3": dt3}
        # info = lambda x: f"{np.mean(x):.2e}+/-{np.std(x):.2e}"
        info = lambda x: f"{np.mean(x):.2e}"
        print(
            f"n_particles = {n:>10}:   dt_v1 = {info(dt1)}s, dt_v2 = {info(dt2)}s, dt_v3 = {info(dt3)}s"
        )

    v1_mean = np.array([np.mean(v["v1"]) for v in results.values()])
    v1_std = np.array([np.std(v["v1"]) for v in results.values()])
    v2_mean = np.array([np.mean(v["v2"]) for v in results.values()])
    v2_std = np.array([np.std(v["v2"]) for v in results.values()])
    v3_mean = np.array([np.mean(v["v3"]) for v in results.values()])
    v3_std = np.array([np.std(v["v3"]) for v in results.values()])

    fig, axs = plt.subplots(
        2,
        1,
        figsize=(5, 3),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0},
    )
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].plot(n_particles, v1_mean, label="v1 (default)", color="r")
    axs[0].plot(n_particles, v2_mean, label="v2 (masking)", color="g")
    axs[0].plot(n_particles, v3_mean, label="v2 compiled (masking)", color="b")
    axs[0].fill_between(
        n_particles, v1_mean + v1_std, v1_mean - v1_std, color="r", alpha=0.3
    )
    axs[0].fill_between(
        n_particles, v2_mean + v2_std, v2_mean - v2_std, color="g", alpha=0.3
    )
    axs[0].fill_between(
        n_particles, v3_mean + v3_std, v3_mean - v3_std, color="b", alpha=0.3
    )
    axs[0].legend()
    axs[0].set_ylabel(r"Time $t$ [s]")
    axs[0].set_xlabel("Number of particles")

    axs[1].set_yscale("log")
    axs[1].plot(n_particles, 1 + 0 * n_particles, "r--")
    axs[1].plot(n_particles, v2_mean / v1_mean, color="g")
    axs[1].plot(n_particles, v3_mean / v1_mean, color="b")
    axs[1].set_ylabel(r"$t_{v2}/t_{v1}$")

    fig.suptitle(f"Time benchmarking for linear layer on {DEVICE}")
    full_str = "SO+13" if USE_FULLY_CONNECTED_SUBGROUP else "O13"
    fig.savefig(f"benchmark_{DEVICE}_{full_str}.pdf", bbox_inches="tight")
    plt.close()
