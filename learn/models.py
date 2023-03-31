import torch
import pytorch_lightning as pl
from torch_geometric.nn import GAT


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