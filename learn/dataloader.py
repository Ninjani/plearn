from pathlib import Path
import torch
from torch.nn.functional import one_hot
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric import transforms as T
import pytorch_lightning as pl
import typing as ty
from tqdm import tqdm

from learn import utils


def load_protein_as_graph(protein_file):
    """
    Load a protein structure from a CIF file and convert it to a PyTorch Geometric Data object 
    The node features are defined using one-hot encodings of element type and amino acid
    `pos` is a tensor of shape (N, 3) where N is the number of atoms, storing the coordinates of each atom
    `y` is a tensor of shape (N, 4) where N is the number of atoms, storing the torsion angles for each residue (repeated for each atom in the residue)
    `chi_indices` is a tensor of shape (R, 4, 4) where R is the number of residues and, for each residue, the indices of the atoms defining the torsion angle are stored for each torsion angle
    `residue_indices` is a tensor of shape N where N is the number of atoms and stores the residue index for each atom (repeated for each atom in the residue)
    `mask` is a boolean tensor of shape (N, 4) indicating which torsion angles are valid
    Parameters
    ----------
    protein_file : str
        The path to the CIF file containing the protein structure
    Returns
    -------
    data : torch_geometric.data.Data
    """
    df = utils.load_protein_as_dataframe(protein_file=protein_file)
    atom_names, residue_names = list(df["atom_name"]), list(df["residue_name"])
    coords = torch.tensor(df[["x_coord", "y_coord", "z_coord"]].values, dtype=torch.float)
    residue_indices = torch.tensor(df["residue_index"].values, dtype=torch.long)
    residue_ends = torch.where(torch.diff(residue_indices) != 0)[0] + 1
    residue_ends = torch.cat([residue_ends, torch.tensor([len(residue_indices) - 1])])

    chi_angles, chi_indices = utils.calculate_torsion_angles(atom_names, residue_names, coords, residue_indices, residue_ends)

    x = torch.cat([one_hot(torch.tensor([utils.ELEMENTS.get(element, len(utils.ELEMENTS)) for element in df["element_symbol"]]),
                                                    num_classes=len(utils.ELEMENTS) + 1).float(),
                one_hot(torch.tensor([utils.AMINO_ACIDS.get(residue_name, len(utils.AMINO_ACIDS)) for residue_name in residue_names]),
                                                    num_classes=len(utils.AMINO_ACIDS) + 1).float()], dim=1)

    data = Data(
        x=x,
        pos=coords,
        y=torch.nan_to_num(chi_angles),
        chi_indices=chi_indices,
        residue_ends=residue_ends,
        n_residues=len(residue_ends),
        mask=~torch.isnan(chi_angles)
    )
    return data


class ProteinDataset(InMemoryDataset):
    """
    A PyTorch Geometric Dataset for proteins
    Takes a file containing a list of proteins to load
    Loads the corresponding protein structures with Pandas and converts them to PyTorch Geometric Data objects using the `load_protein_as_graph` function
    The edges are defined using RadiusGraph with a radius of 8.0 Angstrom
    The edge features are defined using Distance
    The node features are defined using LocalDegreeProfile
    """
    def __init__(self, dataset_type:str, root: str, proteins_list_file: str,
                 num_workers: int = 8, transform: ty.Optional[ty.Callable] = None):
        """
        
        Parameters
        ----------
        dataset_type : str
            The type of dataset, either "train", "val" or "test"
        root : str
            The root directory of the dataset
        proteins_list_file : str
            A file containing a list of proteins to load
        transform : torch_geometric.transforms
            The transforms to apply to the data
        num_workers : int, optional
            The number of workers to use for multiprocessing, by default 8
        """
        self.dataset_type = dataset_type
        self.num_workers = num_workers
        self.protein_names = []
        self.protein_files = []
        with open(proteins_list_file, "r") as f:
            for i, line in enumerate(f):
                if i > 128:
                    break
                protein_file = line.strip()
                self.protein_files.append(protein_file)
                self.protein_names.append(Path(protein_file).stem.split(".")[0].lower())
        self.proteins_list_file = proteins_list_file
        super(ProteinDataset, self).__init__(root, transform=transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return [f"{protein_name}.pt" for protein_name in self.protein_names]

    def download(self):
        with torch.multiprocessing.Pool(self.num_workers) as pool:
            results = [pool.apply_async(self._download_data, 
                                        args=(protein_name, 
                                                protein_file)) for protein_name, protein_file in zip(self.protein_names,
                                                                                                    self.protein_files)]
            for result in tqdm(results, total=len(results)):
                result.wait()
    
    @property
    def processed_file_names(self):
        return [f"data_{self.dataset_type}.pt"]

    def process(self):
        data_list = [torch.load(Path(self.raw_dir) / raw_path) for raw_path in tqdm(self.raw_file_names)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
    def _download_data(self, protein_name, protein_file):
        data_file = Path(self.raw_dir) / f"{protein_name}.pt"
        if not data_file.exists():
            data = load_protein_as_graph(protein_file)
            torch.save(data, data_file)


class ProteinDataModule(pl.LightningDataModule):
    def __init__(self, root, batch_size: int = 32, num_workers: int = 1):
        super().__init__()
        self.train_file = Path(root) / "processed" / "data_train.pt"
        self.val_file = Path(root) / "processed" / "data_val.pt"
        self.test_file = Path(root) / "processed" / "data_test.pt"
        self.train_proteins_list_file = Path(root) / "train.txt"
        self.val_proteins_list_file = Path(root) / "val.txt"
        self.test_proteins_list_file = Path(root) / "test.txt"
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = T.Compose([T.RadiusGraph(8.0), T.Distance(), T.LocalDegreeProfile()])

        if not self.train_file.exists() or not self.val_file.exists():
            self.prepare_data()

    def prepare_data(self) -> None:
        ProteinDataset(dataset_type="train", root=self.root, 
                       proteins_list_file=self.train_proteins_list_file, num_workers=self.num_workers)
        ProteinDataset(dataset_type="val", root=self.root, 
                       proteins_list_file=self.val_proteins_list_file, num_workers=self.num_workers)
        if self.test_proteins_list_file.exists():
            ProteinDataset(dataset_type="test", root=self.root, proteins_list_file=self.test_proteins_list_file, num_workers=self.num_workers)


    def setup(self, stage: ty.Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = ProteinDataset(dataset_type="train", root=self.root, 
                       proteins_list_file=self.train_proteins_list_file, num_workers=self.num_workers, transform=self.transform)
            self.val_dataset = ProteinDataset(dataset_type="val", root=self.root, 
                       proteins_list_file=self.val_proteins_list_file, num_workers=self.num_workers, transform=self.transform)
        elif stage == "test":
            self.test_dataset = ProteinDataset(dataset_type="test", root=self.root, 
                       proteins_list_file=self.test_proteins_list_file, num_workers=self.num_workers, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)