import math
import numpy as np
import torch
import pytorch_lightning as pl
from torch_geometric.nn import GAT
from egnn_pytorch import EGNN_Sparse
from torch.nn import Transformer

from learn.utils import get_rotation


def circular_mse_loss(y_pred, y_true, mask):
    """
    Calculate the circular mean squared error (MSE) between the target values of the 4 angles 'y_true' and
    predicted values of the 4 angles 'y_pred', where mask tells you which angles are present in each residue.
    Args:
    y_true (torch.Tensor): Tensor of shape (num_residues, 4) containing the target values for each residue.
    y_pred (torch.Tensor): Tensor of shape (num_residues, 4) containing the predicted values for each residue.
    mask (torch.Tensor): Tensor of shape (num_residues, 4) containing the mask for each residue.
    Returns:
    (torch.Tensor): Scalar tensor containing the circular MSE loss.
    """
    y_true = torch.nan_to_num(y_true)
    diff = torch.fmod((y_true - y_pred) + torch.pi, 2 * torch.pi) - torch.pi
    ones = torch.full((4,), 1.).to(y_true.device)
    sums = torch.sum(mask, dim=0).float()
    return torch.mean(torch.sum(diff ** 2 * mask.float(), dim=0) / torch.maximum(ones, sums))

class GATModel(pl.LightningModule):
    def __init__(self, in_channels, hidden_channels, num_layers, num_heads, out_channels, dropout, jk):
        super(GATModel, self).__init__()
        self.save_hyperparameters()
        self.example_input_array = (torch.empty(32, in_channels), torch.zeros(2, 100, dtype=torch.long))
        self.model = GAT(in_channels=in_channels,
                         hidden_channels=hidden_channels,
                         num_layers=num_layers,
                         heads=num_heads,
                         out_channels=out_channels,
                         dropout=dropout,
                         jk=jk, v2=True)
        self.validation_step_outputs = []

    def forward(self, node_attributes, edge_index):
        return self.model(node_attributes, edge_index)
    
    def training_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index)
        loss = circular_mse_loss(out, batch.y, batch.mask)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True,
                 batch_size=batch.x.shape[0])
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index)
        loss = circular_mse_loss(out, batch.y, batch.mask)
        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True,
                 batch_size=batch.x.shape[0])
        self.validation_step_outputs.append(dict(loss=loss, y=batch.y, out=out, mask=batch.mask))
        return loss


class EGNNModel(pl.LightningModule):
    def __init__(self, in_channels, num_layers, out_channels):
        super(EGNNModel, self).__init__()
        self.save_hyperparameters()
        self.example_input_array = (torch.empty(32, in_channels), torch.empty(32, 3), torch.zeros(2, 320, dtype=torch.long))
        self.egnn_layers = [EGNN_Sparse(feats_dim=in_channels) for _ in range(num_layers)]
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_channels, out_features=in_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=in_channels, out_features=in_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=in_channels, out_features=out_channels)
            )
        self.validation_step_outputs = []

    def forward(self, node_attributes, pos, edge_index):
        x = torch.cat([pos, node_attributes], dim=-1)
        for layer in self.egnn_layers:
            x = layer(x, edge_index)
        return self.model(x[:, 3:])
    
    def training_step(self, batch, batch_idx):
        out = self(batch.x, batch.pos, batch.edge_index)
        loss = circular_mse_loss(out, batch.y, batch.mask)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True,
                 batch_size=batch.x.shape[0])
        return loss
    
    def validation_step(self, batch, batch_idx):
        out = self(batch.x, batch.pos, batch.edge_index)
        loss = circular_mse_loss(out, batch.y, batch.mask)
        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True,
                 batch_size=batch.x.shape[0])
        self.validation_step_outputs.append(dict(loss=loss, y=batch.y, out=out, mask=batch.mask))
        return loss


class DenseAttention(pl.LightningModule):
    def __init__(self, in_channels, num_heads, out_channels, dropout):
        super(DenseAttention, self).__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.empty(32, in_channels)
        self.model = Transformer(in_channels, num_heads, dropout=dropout)
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_channels, out_features=in_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=in_channels, out_features=in_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=in_channels, out_features=out_channels)
            )
        self.validation_step_outputs = []

    def forward(self, node_attributes):
        return self.output_layer(self.model(node_attributes, node_attributes)[0])
    
    def training_step(self, batch, batch_idx):
        out = self(batch.x)
        loss = circular_mse_loss(out, batch.y, batch.mask)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True,
                 batch_size=batch.x.shape[0])
        return loss
    
    def validation_step(self, batch, batch_idx):
        out = self(batch.x)
        loss = circular_mse_loss(out, batch.y, batch.mask)
        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True,
                 batch_size=batch.x.shape[0])
        self.validation_step_outputs.append(dict(loss=loss, y=batch.y, out=out, mask=batch.mask))
        return loss


class DiffusionDenoiser(pl.LightningModule):
    def __init__(self, in_channels, hidden_channels, num_layers, num_heads, out_channels, dropout, jk):
        super(DiffusionDenoiser, self).__init__()
        self.timestep_embedder = SinusoidalPositionEmbeddings(32)
        self.model = GAT(in_channels=in_channels + 32,
                         hidden_channels=hidden_channels,
                         num_layers=num_layers,
                         heads=num_heads,
                         out_channels=out_channels,
                         dropout=dropout,
                         jk=jk, v2=True)

    def forward(self, node_features, edge_index, batch, timesteps):
        timestep_encoding = torch.repeat_interleave(
            self.timestep_embedder(timesteps), torch.bincount(batch), dim=0)
        x = torch.cat([node_features, timestep_encoding], dim=1)
        x_out = self.model(x, edge_index)
        return x_out


def modify_graph_data(graph, angles):
    angles = torch.fmod(angles, 2 * torch.pi)
    angles = torch.remainder(angles + torch.pi, 2 * torch.pi) - torch.pi
    dihedral_vector_coords = graph.pos[graph.chi_indices]
    start = 0
    for i, end in enumerate(graph.residue_ends):
        for chi in range(4):
            if graph.mask[start, chi]:
                rotation = get_rotation(*dihedral_vector_coords[i, chi], angles[start, chi])
                graph.pos[graph.chi_indices[i, chi, 2]:end] = (rotation @ graph.pos[graph.chi_indices[i, chi, 2]:end].T).T
        start = end
    graph.y = angles
    return graph


class Diffusion(pl.LightningModule):

    def __init__(self, in_channels, hidden_channels, num_layers, num_heads, out_channels, dropout, jk, num_timesteps, diffusion_schedule):
        """
        Initializes the Diffusion class with the given configuration

        Args:
        config: Configuration dictionary containing the parameters for the diffusion model.
        """
        super(Diffusion, self).__init__()
        self.save_hyperparameters()
        # self.example_input_array = (torch.empty(32, in_channels), torch.zeros(2, 100, dtype=torch.long), torch.zeros(32, dtype=torch.long), torch.zeros(32, dtype=torch.long))
        self.model = DiffusionDenoiser(in_channels, hidden_channels, num_layers, num_heads, out_channels, dropout, jk)
        self.validation_step_outputs = []
        self.betas = get_betas(num_timesteps, diffusion_schedule)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat([torch.Tensor([1.]), self.alphas_cumprod[:-1]])

        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)


    def forward(self, node_features, edge_index, batch, timesteps):
        predicted_noise = self.model(node_features, edge_index, batch, timesteps)
        return predicted_noise
    
    def training_step(self, batch, batch_idx):
        """
        Performs a training step for the given batch of graphs.
        Args:
            batch: batch of ground truth graphs

        Returns:
            torch.Tensor: The mean loss for the given batch of graphs.
        """
        timesteps = self.sample_timesteps(len(batch))
        data_sample, noise = self.q_sample(batch, timesteps)
        predicted_noise = self(data_sample.x, data_sample.edge_index, data_sample.batch, timesteps)
        loss = circular_mse_loss(predicted_noise, noise, batch.mask)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True,
                 batch_size=len(batch))
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Performs a validation step for the given batch of graphs.
        Args:
            batch: batch of graphs

        Returns:
            torch.Tensor: The mean loss for the given batch of graphs.
        """
        timesteps = self.sample_timesteps(len(batch))
        data_sample, noise = self.q_sample(batch, timesteps)
        predicted_noise = self(data_sample.x, data_sample.edge_index, data_sample.batch, timesteps)
        loss = circular_mse_loss(predicted_noise, noise, batch.mask)
        predicted_angles = self.p_sample_loop(
            batch.clone(), linspace=self.hparams.num_timesteps)[-1]
        loss_inference = circular_mse_loss(predicted_angles, batch.y, batch.mask)
        self.log('avg_val_loss', loss, on_step=True, on_epoch=True,
                 sync_dist=True, batch_size=batch.x.shape[0])
        self.log('avg_val_loss_inference', loss_inference, on_step=True, sync_dist=True, on_epoch=True,
                 batch_size=batch.x.shape[0])
        self.validation_step_outputs.append(dict(loss=loss_inference, y=batch.y, 
                                                 out=torch.rad2deg(torch.abs(batch.y - predicted_angles)), mask=batch.mask))
        return loss

    def sample_timesteps(self, num_samples):
        """Returns a tensor of random timesteps.

        Args:
            num_samples (int): The number of samples.

        Returns:
            torch.Tensor: A 1D tensor of random timesteps.
        """
        return torch.randint(0, self.hparams.num_timesteps, size=(num_samples,)).to(self.device)
    
    
    def q_sample(self, data, timestep):
        """
        Samples a new torsion configuration by adding random noise to the current configuration.

        Args:
            data (Data): protein graph with ground truth torsion angles stored in y.
            timestep (Tensor): The timestep to sample

        Returns:
            Tuple[Data, Tensor]: A tuple containing the protein graph with the new torsions
            and the noise added to it.
        """
        noise = torch.rand_like(data.y) * 2 * torch.pi - torch.pi
        new_angles = extract(self.sqrt_alphas_cumprod, timestep, data.batch) * data.y + extract(self.sqrt_one_minus_alphas_cumprod, timestep, data.batch) * noise
        data_sample = modify_graph_data(data.clone(), new_angles)
        return data_sample, noise

    @torch.no_grad()
    def p_sample(self, data, timestep):
        sqrt_recip_alphas_t = extract((1. / self.sqrt_alphas), timestep,
                                      data.batch)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, timestep,
                                                  data.batch)
        betas_t = extract(self.betas, timestep, data.batch)
        w_noise = betas_t / sqrt_one_minus_alphas_cumprod_t
        # Use our model (noise predictor) to predict the mean
        predicted_noise = self.model(data.x, data.edge_index, data.batch,
                                     timestep)
        model_mean = sqrt_recip_alphas_t * (data.y - w_noise * predicted_noise)
        data = modify_graph_data(data, model_mean)
        if timestep[0] != 0:
            posterior_variance_t = extract(self.posterior_variance, timestep, data.batch)
            noise = torch.randn_like(data.y) * 2 * torch.pi - torch.pi
            data = modify_graph_data(data, data.y + posterior_variance_t * noise)
        return data

    @torch.no_grad()
    def p_sample_loop(self, batch, noise=True, linspace=1000, start_timestep=None):
        if start_timestep is None:
            start_timestep = self.hparams.num_timesteps - 1
        start_timestep = min(
            start_timestep, self.hparams.num_timesteps - 1)
        linspace = min(linspace, start_timestep)
        if noise:
            # start from pure noise (for each example in the batch)
            batch.y = torch.rand_like(batch.y) * 2 * torch.pi - torch.pi
        angles = []
        for timestep in reversed(np.linspace(0, start_timestep, linspace).astype(int)):
            timesteps = torch.full((len(batch),), timestep).to(self.device)
            batch = self.p_sample(batch, timesteps)
            angles.append(batch.y.detach())
        return angles
    

def extract(a, t, batch_idx):
    """
    Extract the necessary alphas or betas from the input array 'a' corresponding to the time steps 't'
    for each molecule in 'batch_idx'.

    Args:
    a (torch.Tensor): Array containing the alphas or betas.
    t (torch.Tensor): Tensor of shape (batch_size,) containing the time step for each entry.
    batch_idx (torch.Tensor): Tensor of shape (batch_size * num_nodes,) containing the batch index of each node in each graph.

    Returns:
    (torch.Tensor): Array of shape (batch_size * num_nodes, 1) containing the extracted alphas or betas.
    """
    return torch.repeat_interleave(a.to(t.device)[t], torch.bincount(batch_idx.to(t.device))).reshape(
        (len(batch_idx.to(t.device)), 1)).to(
        t.device)


def get_betas(n_timestep, schedule):
    if schedule == 'linear':
        return torch.linspace(0.0001, 0.02, n_timestep)
    elif schedule == 'cosine':
        steps = n_timestep + 1
        x = torch.linspace(0, n_timestep, steps)
        alphas_cumprod = torch.cos((x / steps) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    else:
        print('Invalid schedule: {}'.format(schedule))
        exit(0)

class SinusoidalPositionEmbeddings(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings