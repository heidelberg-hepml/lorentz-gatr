import numpy as np

from experiments.eventgen.experiment import EventGenerationExperiment


class zmumuExperiment(EventGenerationExperiment):
    def define_process_specifics(self):
        self.plot_title = r"Z"
        self.n_hard_particles = 2
        self.n_jets_max = 5
        self.onshell_list = [0, 1]
        self.onshell_mass = [0.1, 0.1]
        self.units = 258.1108
        self.base_kwargs = {
            "pxy_std": 36.6,
            "pz_std": 368.7,
            "logpt_mean": 3.61,
            "logpt_std": 0.56,
            "logmass_mean": 1.08,
            "logmass_std": 1.24,
            "eta_std": 1.98,
        }
        self.delta_r_min = 0.39
        self.pt_min = [0.0, 0.0, 20.0, 20.0, 20.0]
        self.obs_names_index = ["l1", "l2"]
        for ijet in range(self.n_jets_max):
            self.obs_names_index.append(f"j{ijet+1}")
        self.fourmomentum_ranges = [[0, 200], [-150, 150], [-150, 150], [-150, 150]]
        self.jetmomentum_ranges = [[10, 150], [-np.pi, np.pi], [-6, 6], [0, 20]]
        self.virtual_components = [[0, 1]]
        self.virtual_ranges = [[0, 300], [-np.pi, np.pi], [-6, 6], [75, 115]]
        self.virtual_names = [
            r"p_{T,\mu\mu}",
            r"\phi_{\mu\mu}",
            r"\eta_{\mu\mu}",
            r"m_{\mu\mu}",
        ]
