import numpy as np

from experiments.eventgen.experiment import EventGenerationExperiment


class zmumuExperiment(EventGenerationExperiment):
    def define_process_specifics(self):
        self.plot_title = r"Z"
        self.n_hard_particles = 2
        self.n_jets_max = 5
        self.is_onshell = [0, 1]
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
