import numpy as np


def inner_p(p1, p2):
    return np.log(
        p1[:, 0] * p2[:, 0]
        - p1[:, 1] * p2[:, 1]
        - p1[:, 2] * p2[:, 2]
        - p1[:, 3] * p2[:, 3]
    )


def preprocess_particles(particles_raw, mean=None, std=None, eps_std=1e-2):
    if mean is None or std is None:
        mean = particles_raw.mean((0, 1))
        std = particles_raw.std((0, 1))
        std = np.clip(std, a_min=eps_std, a_max=None)  # avoid std=0.

    particles = (particles_raw - mean) / std
    assert np.isfinite(particles).all()
    return particles, mean, std


def preprocess_particles_w_invariants(particles_raw, mean=None, std=None, eps_std=1e-2):
    p_grouped = np.transpose(particles_raw, (1, 0, 2))
    p_single_array = particles_raw.reshape(particles_raw.shape[0], -1)
    for i in range(p_grouped.shape[0]):
        for j in range(i + 1, p_grouped.shape[0]):
            p_single_array = np.concatenate(
                (p_single_array, inner_p(p_grouped[i], p_grouped[j])[:, None]), axis=1
            )

    mean = p_single_array.mean((0))
    std = p_single_array.std((0))
    std = np.clip(std, a_min=eps_std, a_max=None)

    p_single_array_prepd = (p_single_array - mean) / std

    return p_single_array_prepd


def preprocess_amplitude(amplitude, std=None):
    log_amplitude = np.log(amplitude)
    if std is None:
        mean = log_amplitude.mean()
        std = log_amplitude.std()
    prepd_amplitude = (log_amplitude - mean) / std
    assert np.isfinite(prepd_amplitude).all()
    return prepd_amplitude, mean, std


def undo_preprocess_amplitude(prepd_amplitude, mean, std):
    assert mean is not None and std is not None
    log_amplitude = prepd_amplitude * std + mean
    amplitude = np.exp(log_amplitude)
    return amplitude
