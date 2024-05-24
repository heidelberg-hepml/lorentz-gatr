import numpy as np

from experiments.eventgen.experiment import EventGenerationExperiment


class ttbarExperiment(EventGenerationExperiment):
    '''
    Process: p p > t t~ at reco-level with hadronic top decays and 0-4 extra jets
    Main experiment, used for the paper
    Dataset will be published at the ITP website
    '''
    def define_process_specifics(self):
        self.plot_title = r"t\bar t"
        self.n_hard_particles = 6
        self.n_jets_max = 4
        self.onshell_list = []
        self.onshell_mass = []
        self.units = 206.6
        self.pt_min = [22.0] * 10
        self.delta_r_min = 0.5
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

class zmumuExperiment(EventGenerationExperiment):
    '''
    Process: p p > z > mu+ mu- at reco-level with 1-3 extra jets
    For comparison with our previous paper https://arxiv.org/abs/2305.10475
    Dataset available upon request
    '''
    def define_process_specifics(self):
        self.plot_title = r"Z"
        self.n_hard_particles = 2
        self.n_jets_max = 5
        self.onshell_list = [0, 1]
        self.onshell_mass = [0.1, 0.1]
        self.units = 258.1108
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

class z5gExperiment(EventGenerationExperiment):
    '''
    Process: p p > z g g g g g at parton level
    Enhance statistics on Z+5g dataset for amplitude regression
    Dataset available upon request
    '''
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
