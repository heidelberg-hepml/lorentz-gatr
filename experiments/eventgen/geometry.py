from experiments.eventgen.helpers import ensure_angle


class BaseGeometry:
    def get_trajectory(self, x_target, x_base, t):
        raise NotImplementedError

    def get_metric(self, y1, y2, x):
        raise NotImplementedError


class SimpleGeometry:
    def get_trajectory(self, x_target, x_base, t):
        v_t = x_base - x_target
        x_t = x_target + t * v_t
        return x_t, v_t

    def get_metric(self, y1, y2, x):
        # y1 and y2 are vectors (not necessarily positions), and x is the position
        # default: euclidean metric
        se = (y1 - y2) ** 2 / 2
        return se.mean(dim=[-1, -2])


class SimplePossiblyPeriodicGeometry(SimpleGeometry):
    def __init__(self, contains_phi=False, periodic=True):
        self.periodic_components = [1] if contains_phi and periodic else []

    def _handle_periodic(self, x):
        x[..., self.periodic_components] = ensure_angle(
            x[..., self.periodic_components]
        )
        return x

    def get_trajectory(self, x_target, x_base, t):
        v_t = x_base - x_target
        v_t = self._handle_periodic(v_t)
        x_t = x_target + t * v_t
        x_t = self._handle_periodic(x_t)
        return x_t, v_t

    def get_metric(self, y1, y2, x):
        diff = y1 - y2
        # diff = self._handle_periodic(diff)
        se = diff**2 / 2
        return se.mean(dim=[-1, -2])
