import torch
from torch import nn
from torch_geometric.utils import dense_to_sparse

from gatr.interface import embed_vector, extract_scalar


def encode_tokens(type_token, global_token, token_size, isgatr, batchsize, device):
    """Compute embedded type_token and global_token to be used within Transformers

    Parameters
    type_token: iterable of int
        list with type_tokens for each particle in the event
    global_token: int
    isgatr: bool
        whether the encoded tokens will be used within L-GATr or within the baseline Transformer
        This affects how many zeroes have to be padded to the global_token (4 more for the baseline Transformer)
    batchsize: int
    device: torch.device


    Returns:
    type_token: torch.Tensor with shape (batchsize, num_particles, type_token_max)
        one-hot-encoded type tokens, to be appended to each encoded 4-momenta in case of the
        baseline transformer / make up the full scalar channel for L-GATr
    global_token: torch.Tensor with shape (batchsize, 1, type_token_max+4)
        ont-hot-encoded dataset token, this will be the global_token and appended to the individual particles
    """
    type_token = nn.functional.one_hot(type_token, num_classes=token_size)
    type_token = (
        type_token.unsqueeze(1)
        .expand(type_token.shape[0], batchsize, *type_token.shape[1:])
        .float()
    )

    global_token = nn.functional.one_hot(
        global_token, num_classes=token_size + (0 if isgatr else 4)
    )
    global_token = (
        global_token.unsqueeze(1)
        .expand(global_token.shape[0], batchsize, *global_token.shape[1:])
        .unsqueeze(-2)
        .float()
    )
    return type_token, global_token


class AmplitudeMLPWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs, type_token, global_token, attn_mask=None):
        # ignore type_token, global_token and attn_mask (architecture is not permutation invariant)
        nprocesses, batchsize, num_particles, num_components = inputs.shape
        inputs = inputs.reshape(nprocesses, batchsize, num_particles * num_components)
        out = self.net(inputs)
        return out


class AmplitudeDSIWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs, type_token, global_token, attn_mask=None):
        out = self.net(inputs, type_token)
        return out


class AmplitudeTransformerWrapper(nn.Module):
    def __init__(self, net, token_size):
        super().__init__()
        self.net = net
        self.token_size = token_size

    def forward(self, inputs, type_token, global_token, attn_mask=None):
        nprocesses, batchsize, _, _ = inputs.shape

        type_token, global_token = encode_tokens(
            type_token,
            global_token,
            self.token_size,
            isgatr=False,
            batchsize=batchsize,
            device=inputs.device,
        )

        # type_token
        inputs = torch.cat((inputs, type_token), dim=-1)

        # global_token
        inputs = torch.cat((global_token, inputs), dim=-2)

        outputs = self.net(inputs, attention_mask=attn_mask)
        amplitudes = outputs[..., 0, :]

        return amplitudes


class AmplitudeGAPWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs: torch.Tensor, type_token, global_token, attn_mask=None):
        # ignore type token, global_token and attn_mask
        multivector, scalars = self.embed_into_ga(inputs)
        multivector_outputs, scalar_outputs = self.net(multivector, scalars=scalars)
        amplitude = self.extract_from_ga(multivector_outputs, scalar_outputs)
        return amplitude

    def embed_into_ga(self, inputs):
        nprocesses, batchsize, _, _ = inputs.shape

        # encode momenta in multivectors
        multivector = embed_vector(inputs)
        scalars = torch.zeros((nprocesses, batchsize, 1), device=inputs.device)

        return multivector, scalars

    def extract_from_ga(self, multivector, scalars):
        # Extract scalars from GA representation
        outputs = extract_scalar(multivector)[..., 0]

        return outputs


class AmplitudeGATrWrapper(nn.Module):
    def __init__(self, net, token_size, reinsert_type_token=False):
        super().__init__()
        self.net = net
        self.token_size = token_size

    def forward(self, inputs: torch.Tensor, type_token, global_token, attn_mask=None):
        multivector, scalars = self.embed_into_ga(inputs, type_token, global_token)
        multivector_outputs, scalar_outputs = self.net(
            multivector, scalars=scalars, attention_mask=attn_mask
        )
        amplitude = self.extract_from_ga(multivector_outputs, scalar_outputs)
        return amplitude

    def embed_into_ga(self, inputs, type_token, global_token):
        nprocesses, batchsize, num_objects, _ = inputs.shape

        # encode momenta in multivectors
        multivector = embed_vector(inputs)
        multivector = multivector.unsqueeze(-2)

        type_token, global_token = encode_tokens(
            type_token,
            global_token,
            self.token_size,
            isgatr=True,
            batchsize=batchsize,
            device=inputs.device,
        )
        type_token = type_token.to(inputs.dtype)
        global_token = global_token.to(inputs.dtype)

        # encode type_token in scalars
        scalars = type_token

        # global token
        global_token_mv = torch.zeros(
            (nprocesses, batchsize, 1, multivector.shape[-2], multivector.shape[-1]),
            dtype=multivector.dtype,
            device=multivector.device,
        )
        global_token_s = global_token
        multivector = torch.cat((global_token_mv, multivector), dim=-3)
        scalars = torch.cat((global_token_s, scalars), dim=-2)
        return multivector, scalars

    def extract_from_ga(self, multivector, scalars):
        # Extract scalars from GA representation
        lorentz_scalars = extract_scalar(multivector)[..., 0]

        amplitude = lorentz_scalars[..., 0, :]
        return amplitude


class AmplitudeCGENNWrapper(nn.Module):
    def __init__(self, net, token_size):
        super().__init__()
        self.net = net
        self.token_size = token_size

    def embed_into_ga(self, inputs, type_token, global_token):
        # very similar to GATrWrapper
        nprocesses, batchsize, num_objects, _ = inputs.shape

        # encode momenta in multivectors
        multivector = embed_vector(inputs)
        multivector = multivector.unsqueeze(-2)

        type_token, global_token = encode_tokens(
            type_token,
            global_token,
            self.token_size,
            isgatr=True,
            batchsize=batchsize,
            device=inputs.device,
        )

        # encode type_token in scalars
        scalars = type_token
        return multivector, scalars

    def forward(self, inputs, type_token, global_token, attn_mask=None):
        # CGENN does not support multiple batch dimensions
        assert (
            len(inputs.shape) == 4
        ), "The CGENN implementation does not support joint training yet"

        multivectors, scalars = self.embed_into_ga(inputs, type_token, global_token)

        # convert to sparse tensors
        multivectors = multivectors.view(
            -1, multivectors.shape[-2], multivectors.shape[-1]
        )
        scalars = scalars.view(-1, scalars.shape[-1])

        # edge_index of fully connected graph
        adj_matrix = torch.ones(
            (inputs.shape[-2], inputs.shape[-2]), device=inputs.device
        )
        edge_index_single = dense_to_sparse(adj_matrix)[0]
        edge_index = torch.cat(
            ([edge_index_single + inputs.shape[2] * i for i in range(inputs.shape[1])]),
            dim=-1,
        )

        # construct edge and node features
        i, j = edge_index
        edge_attr_x = torch.cat(
            (multivectors[i], multivectors[j], multivectors[i] - multivectors[j]),
            dim=-2,
        )
        node_attr_x = multivectors
        node_attr_h = scalars

        # pass everything through the network
        out = self.net(
            x=multivectors,
            h=scalars,
            edge_attr_x=edge_attr_x,
            edge_attr_h=None,
            node_attr_x=node_attr_x,
            node_attr_h=node_attr_h,
            edges=edge_index,
            n_nodes=inputs.shape[-2],
        )
        return out[None, ...]
