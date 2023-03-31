from pytorch_lightning.cli import LightningCLI
import torch

from learn.dataloader import ProteinDataModule

def main():
    """
    Run with python main.py fit -c config.yaml
    Or in an sbatch script with srun python main.py fit -c config.yaml
    """
    torch.set_float32_matmul_precision('medium')
    cli = LightningCLI(datamodule_class=ProteinDataModule)

if __name__ == '__main__':
    main()
