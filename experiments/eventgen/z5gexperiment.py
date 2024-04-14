import numpy as np

from experiments.eventgen.experiment import EventGenerationExperiment


class z5gExperiment(EventGenerationExperiment):
    def define_process_specifics(self):
        self.plot_title = r"Z"
        self.n_hard_particles = 1
        self.n_jets_max = 5
        self.onshell_list = [0, 1, 2, 3, 4, 5]
        self.onshell_mass = [91.188, 0.1, 0.1, 0.1, 0.1, 0.1]
        self.units = 275.69
        self.delta_r_min = 0.4
        self.pt_min = [0.0, 20.0, 20.0, 20.0, 20.0, 20.0]
        self.obs_names_index = ["Z"]
        self.obs_names_index.extend([f"g{i}" for i in range(1, 6)])
        self.fourmomentum_ranges = [[0, 400], [-150, 150], [-150, 150], [-150, 150]]
        self.jetmomentum_ranges = [[10, 150], [-np.pi, np.pi], [-6, 6], [0, 20]]
        self.virtual_components = []
        self.virtual_ranges = []
        self.virtual_names = []
