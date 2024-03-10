import numpy as np

from experiments.eventgen.experiment import EventGenerationExperiment


class ttbarExperiment(EventGenerationExperiment):
    def define_process_specifics(self):
        self.plot_title = r"t\bar t"
        self.n_hard_particles = 6
        self.n_jets_max = 4
        self.onshell_list = []
        self.onshell_mass = []
        self.pt_min = [21.49] * 10
        self.obs_names_index = ["b1", "q1", "q2", "b2", "q3", "q4"]
        for ijet in range(self.n_jets_max):
            self.obs_names_index.append(f"j{ijet+1}")
        self.fourmomentum_ranges = [[0, 200], [-150, 150], [-150, 150], [-150, 150]]
        self.jetmomentum_ranges = [[10, 150], [-np.pi, np.pi], [-6, 6], [0, 20]]
        self.virtual_components = [
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2],
            [3, 4, 5],
            [1, 2],
            [4, 5],
        ]
        self.virtual_names = [
            r"p_{T,t\bar t}",
            r"\phi_{t\bar t}",
            r"\eta_{t\bar t}",
            r"m_{t\bar t}",
            "p_{T,t}",
            "\phi_t",
            "\eta_t",
            "m_{ t }",
            r"p_{T,\bar t}",
            r"\phi_{\bar t}",
            r"\eta_{\bar t}",
            r"m_{\bar t}",
            "p_{T,W^+}",
            "\phi_{W^+}",
            "\eta_{W^+}",
            "m_{W^+}",
            "p_{T,W^-}",
            "\phi_{W^-}",
            "\eta_{W^-}",
            "m_{W^-}",
        ]
        self.virtual_ranges = [
            [0, 500],
            [-np.pi, np.pi],
            [-6, 6],
            [200, 1000],
            [0, 500],
            [-np.pi, np.pi],
            [-6, 6],
            [50, 400],
            [0, 500],
            [-np.pi, np.pi],
            [-6, 6],
            [50, 400],
            [0, 300],
            [-np.pi, np.pi],
            [-6, 6],
            [30, 150],
            [0, 300],
            [-np.pi, np.pi],
            [-6, 6],
            [30, 150],
        ]
