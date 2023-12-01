import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import make_dataclass
from tqdm import trange

from gatr.nets import GATr
from gatr.layers.attention.config import SelfAttentionConfig
from gatr.layers.mlp.config import MLPConfig
from gatr.interface.vector import embed_vector, extract_vector

def simulate_data(n_events, mass_min=0.01, mass_max=1., p_mu=0., p_std=1.):
    mass = torch.rand(n_events) * (mass_max - mass_min) + mass_min
    p = torch.randn(n_events, 3) * p_std + p_mu
    E = torch.sqrt(mass**2 + torch.sum(p**2, dim=1))
    events = torch.cat((E.unsqueeze(-1), p), dim=1)
    return events, mass

class MassRegressionExperiment:
    '''
    Quick test
    '''
    
    def __init__(self, params):
        self.params = params
        self.n_events = 10000
        self.batch_size = 32
        self.lr=1e-3
        self.epochs = 10

    def full_run(self):
        self.create_dataloader()
        self.create_model()
        self.train_model()
        
    def create_dataloader(self):
        self.events, self.masses = simulate_data(self.n_events)
        dataset = torch.utils.data.TensorDataset(torch.tensor(self.events), torch.tensor(self.masses))
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def create_model(self):
        # TBD: Correctly initialize model
        #mlp_config = dict(mv_channels=[1,4,1], s_channels=[0,0,0], activation="gelu", dropout_prob=0.1)
        #mlp = MLPConfig(*mlp_config)
        mlp = MLPConfig()
        #mlp.mv_channels = [1,1,1,1]
        #mlp.s_channels = [1,1,1,1]
        #attention_config = dict(in_mv_channels=1, out_mv_channels=1,
        #                  in_s_channels=0, out_s_channels=0, 
        #                  num_heads=1)
        #attention = SelfAttentionConfig(*attention_config)
        attention = SelfAttentionConfig()
        config = dict(in_mv_channels=1, out_mv_channels=1, hidden_mv_channels=1,
                          in_s_channels=1, out_s_channels=1, hidden_s_channels=1,
                          num_blocks=1, mlp=mlp, attention=attention)
        net = GATr(**config)
        self.model = Wrapper(net)

    def train_model(self):
        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        for _ in (pbar := trange(self.epochs)):
            for x, y in self.dataloader:
                y_pred = self.model(x)
                loss = loss_fn(y, y_pred)
                optim.zero_grad()
                loss.backward()
                optim.step()
                pbar.set_description(desc=f"Loss = {loss.item():.3f}", refresh=True)

class Wrapper(nn.Module):

    def __init__(self, net, return_other=False):
        super().__init__()
        self.net = net
        self.return_other = return_other

    def forward(self, inputs: torch.Tensor):
        print(inputs.shape)
        mv, s = self.embed_into_ga(inputs)
        print("IN", mv.shape, s.shape)
        mv_outputs, s_outputs = self.net(mv, s)
        print("OUT", mv_outputs.shape, s_outputs.shape)

        outputs, other = self.extract_from_ga(mv_outputs, s_outputs)
        if self.return_other:
            return outputs, other
        else:
            return outputs

    def embed_into_ga(self, inputs):
        inputs = inputs.unsqueeze(1).unsqueeze(1)
        batch_size, num_objects, _, _ = inputs.shape
        print("INPUTS", inputs.shape)
        #inputs = inputs.unsqueeze(-1)

        mv = embed_vector(inputs)
        s = torch.zeros((batch_size, num_objects, 1), device=inputs.device)
        # mv has shape [batchsize, num_objects, num_channels, 16] (this is the shape of things GATr works with)
        # and s has shape [batchsize, num_objects, num_channels]
        
        return mv, s

    def extract_from_ga(self, multivectors, scalars):
        # TBD
        assert multivector.shape[2:] == (1, 16)
        assert scalars.shape[2:] == (1,)

        output = extract_vector(multivector[:,:,0,:])
        
        return output

        
        
        
