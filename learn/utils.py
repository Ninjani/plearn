from biopandas.pdb import PandasPdb
from biopandas.mmcif import PandasMmcif
from pathlib import Path
import typing as ty
from scipy.spatial.transform import Rotation

import torch


ELEMENTS: ty.List[str] = "C H O N F P S Cl Br I B".split() # type: ignore
ELEMENTS: ty.Dict[str, int] = dict(zip(ELEMENTS, range(len(ELEMENTS))))
AMINO_ACIDS: ty.List[str] = "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split() # type: ignore
AMINO_ACIDS: ty.Dict[str, int] = dict(zip(AMINO_ACIDS, range(len(AMINO_ACIDS))))
# Define atoms involved in each chi angle
CHI_ATOMS = {'VAL': {'chi1': ('N', 'CA', 'CB', 'CG1')},
             'LEU': {'chi1': ('N', 'CA', 'CB', 'CG'), 'chi2': ('CA', 'CB', 'CG', 'CD1')},
             'ILE': {'chi1': ('N', 'CA', 'CB', 'CG1'), 'chi2': ('CA', 'CB', 'CG1', 'CD1')},
             'MET': {'chi1': ('N', 'CA', 'CB', 'CG'), 'chi2': ('CA', 'CB', 'CG', 'SD'), 'chi3': ('CB', 'CG', 'SD', 'CE')},
             'PHE': {'chi1': ('N', 'CA', 'CB', 'CG'), 'chi2': ('CA', 'CB', 'CG', 'CD1')},
             'PRO': {'chi1': ('N', 'CA', 'CB', 'CG'), 'chi2': ('CA', 'CB', 'CG', 'CD')},
             'SER': {'chi1': ('N', 'CA', 'CB', 'OG')},
             'THR': {'chi1': ('N', 'CA', 'CB', 'OG1')},
             'CYS': {'chi1': ('N', 'CA', 'CB', 'SG')},
             'ASN': {'chi1': ('N', 'CA', 'CB', 'CG'), 'chi2': ('CA', 'CB', 'CG', 'OD1')},
             'GLN': {'chi1': ('N', 'CA', 'CB', 'CG'), 'chi2': ('CA', 'CB', 'CG', 'CD'), 'chi3': ('CB', 'CG', 'CD', 'OE1')},
             'TYR': {'chi1': ('N', 'CA', 'CB', 'CG'), 'chi2': ('CA', 'CB', 'CG', 'CD1')},
             'TRP': {'chi1': ('N', 'CA', 'CB', 'CG'), 'chi2': ('CA', 'CB', 'CG', 'CD1')},
             'ASP': {'chi1': ('N', 'CA', 'CB', 'CG'), 'chi2': ('CA', 'CB', 'CG', 'OD1')},
             'GLU': {'chi1': ('N', 'CA', 'CB', 'CG'), 'chi2': ('CA', 'CB', 'CG', 'CD'), 'chi3': ('CB', 'CG', 'CD', 'OE1')},
             'HIS': {'chi1': ('N', 'CA', 'CB', 'CG'), 'chi2': ('CA', 'CB', 'CG', 'ND1')},
             'LYS': {'chi1': ('N', 'CA', 'CB', 'CG'), 'chi2': ('CA', 'CB', 'CG', 'CD'), 'chi3': ('CB', 'CG', 'CD', 'CE'), 'chi4': ('CG', 'CD', 'CE', 'NZ')},
             'ARG': {'chi1': ('N', 'CA', 'CB', 'CG'),
                     'chi2': ('CA', 'CB', 'CG', 'CD'),
                     'chi3': ('CB', 'CG', 'CD', 'NE'),
                     'chi4': ('CG', 'CD', 'NE', 'CZ'),
                     'chi5': ('CD', 'NE', 'CZ', 'NH1')}}

PDB_COLUMN_ORDER: ty.List[str] = [
    "record_name",
    "atom_number",
    "blank_1",
    "atom_name",
    "alt_loc",
    "residue_name",
    "blank_2",
    "chain_id",
    "residue_number",
    "insertion",
    "blank_3",
    "x_coord",
    "y_coord",
    "z_coord",
    "occupancy",
    "b_factor",
    "blank_4",
    "segment_id",
    "element_symbol",
    "charge",
    "line_idx",
]

MMCIF_PDB_COLUMN_MAP: ty.Dict[str, str] = {
    "group_PDB": "record_name",
    "id": "atom_number",
    "auth_atom_id": "atom_name",
    "auth_comp_id": "residue_name",
    "auth_asym_id": "chain_id",
    "auth_seq_id": "residue_number",
    "Cartn_x": "x_coord",
    "Cartn_y": "y_coord",
    "Cartn_z": "z_coord",
    "occupancy": "occupancy",
    "B_iso_or_equiv": "b_factor",
    "type_symbol": "element_symbol",
    "pdbx_PDB_model_num": "model_number",
    "pdbx_formal_charge": "charge",
    "pdbx_PDB_ins_code": "insertion"
}

MMCIF_PDB_NONEFIELDS: ty.List[str] = [
    "blank_1",
    "alt_loc",
    "blank_2",
    "blank_3",
    "blank_4",
    "segment_id",
    "line_idx",
]

def guess_protein_type(protein_file: ty.Union[str, Path]):
    """
    Guess the type of a protein file
    Parameters
    ----------
    protein_file: Path
        Path to a protein file
    Returns
    -------
    protein_type: str
        Type of the protein file
    """
    if type(protein_file) is not Path and not Path(protein_file).exists():
        if len(str(protein_file)) == 4:
            return "pdb_id"
        else:
            return "uniprot_id"
    protein_file = Path(protein_file)
    if any(x in protein_file.suffixes for x in [".pdb", ".ent"]):
        protein_type = "pdb"
    elif ".cif" in protein_file.suffixes:
        protein_type = "mmcif"
    else:
        raise ValueError("Unknown protein file type")
    return protein_type


def convert_to_pandas_pdb(mmcif, offset_chains: bool = True, records: ty.List[str] = ["ATOM", "HETATM"]) -> PandasPdb:
    """Returns a PandasPdb object with the same data as the PandasMmcif
    object.
    Attributes
    ----------
    offset_chains: bool
        Whether or not to offset atom numbering based on number of chains.
        This can arise due to the presence of TER records in PDBs which are
        not found in mmCIFs.
    records: List[str]
        List of record types to save. Any of ["ATOM", "HETATM", "OTHERS"].
        Defaults to ["ATOM", "HETATM"].
    """
    pandaspdb = PandasPdb()

    for a in records:
        try:
            dfa = mmcif.df[a]
            # keep only those fields found in pdb
            dfa = dfa[MMCIF_PDB_COLUMN_MAP.keys()]
            # rename fields
            dfa = dfa.rename(columns=MMCIF_PDB_COLUMN_MAP)
            # add empty fields
            for i in MMCIF_PDB_NONEFIELDS:
                dfa[i] = ""
            # reorder columns to PandasPdb order
            dfa = dfa[PDB_COLUMN_ORDER]
            pandaspdb.df[a] = dfa
        except KeyError:  # Some entries may not have an ANISOU
            continue

    # update line_idx
    pandaspdb.df["ATOM"]["line_idx"] = pandaspdb.df["ATOM"].index.values
    pandaspdb.df["HETATM"]["line_idx"] = pandaspdb.df["HETATM"].index

    # Update atom numbers
    if offset_chains:
        offsets = pandaspdb.df["ATOM"]["chain_id"].astype(
            "category").cat.codes
        pandaspdb.df["ATOM"]["atom_number"] = pandaspdb.df["ATOM"]["atom_number"] + offsets
        hetatom_offset = offsets.max() + 1
        pandaspdb.df["HETATM"]["atom_number"] = pandaspdb.df["HETATM"]["atom_number"] + hetatom_offset

    return pandaspdb


def load_protein_as_dataframe(protein_file: ty.Union[Path, str], should_be=None):
    """
    Load a protein as a Pandas DataFrame with the following columns:
    - chain_id
    - residue_number
    - residue_name
    - atom_name
    - x_coord
    - y_coord
    - z_coord
    - occupancy
    - b_factor
    - alt_loc
    - insertion
    - atom_number
    - element_symbol
    - charge
    - node_id

    Parameters
    ----------
    protein_file: Path, str
        Path to a mmCIF file (can be zipped) or
          a PDB file (can be zipped) or
          a PDB ID to fetch from the PDB or
          a Uniprot ID to fetch from the AlphaFold database
    
    Returns
    -------
    df: Pandas DataFrame
    """

    # load pdb
    protein_type = guess_protein_type(protein_file)
    if should_be is not None:
        assert protein_type == should_be, f"Expected {should_be} but got {protein_type} for {protein_file}"
    if protein_type == "pdb":
        df = PandasPdb().read_pdb(str(protein_file))
    elif protein_type == "mmcif":
        df = PandasMmcif().read_mmcif(str(protein_file))
    elif protein_type == "pdb_id":
        df = PandasMmcif().fetch_mmcif(str(protein_file))
    elif protein_type == "uniprot_id":
        df = PandasMmcif().fetch_mmcif(uniprot_id=str(protein_file), source="alphafold-v2")
    else:
        raise ValueError("Unknown protein file type")

    if protein_type != "pdb":
        df = convert_to_pandas_pdb(df)

    df = df.df['ATOM']

    # remove atoms with alternative locations, keep the one with the highest occupancy
    df = df.sort_values("occupancy")
    duplicates = df.duplicated(
        subset=["chain_id", "residue_number", "atom_name", "atom_number"],
        keep="last",
    )
    df = df[~duplicates]

    # remove insertion codes
    df = df[df["insertion"].isin(["", "None", None])]
    df.reset_index(drop=True, inplace=True)

    df.sort_values(
        by=['chain_id', 'residue_number', 'residue_name', 'atom_number'], inplace=True
    )

    df["node_id"] = (
        df["chain_id"].apply(str)
        + ":"
        + df["residue_name"]
        + ":"
        + df["residue_number"].apply(str)
        + ":"
        + df["atom_name"]
    )
    df['residue_index'] = df.groupby(['chain_id', 'residue_number', 'residue_name']).ngroup()

    return df


def calc_dihedral(v1, v2, v3, v4):
    ab, cb, db = v2 - v1, v3 - v2, v4 - v3
    u, v = torch.cross(ab, cb), torch.cross(db, cb)
    w = torch.cross(u, v)
    angle_uv = torch.dot(u, v) / (torch.norm(u, dim=-1) * torch.norm(v, dim=-1))
    angle_cbw = torch.dot(cb, w) / (torch.norm(cb, dim=-1) * torch.norm(w, dim=-1))
    angle = torch.acos(torch.clamp(angle_uv, -1, 1))
    return torch.where(torch.acos(torch.clamp(angle_cbw, -1, 1)) > 0.001, -angle, angle)


def calculate_torsion_angles(atom_names, residue_names, coords, residue_indices, residue_ends):
    chi_angles = torch.full((len(atom_names), 4), float('nan'), dtype=torch.float32)
    chi_indices = torch.zeros((len(residue_ends), 4, 4), dtype=torch.long)
    atom_name_to_index = dict(zip(zip(residue_indices.detach().numpy(), atom_names), range(len(atom_names))))
    start = 0
    for end in residue_ends:
        residue_name, residue_index = residue_names[start], residue_indices[start].item()
        if residue_name in CHI_ATOMS:
            for chi in range(4):
                chi_atoms = CHI_ATOMS[residue_name].get(f"chi{chi+1}")
                if chi_atoms and all((residue_index, atom_name) in atom_name_to_index for atom_name in chi_atoms):
                    chi_atom_indices = [atom_name_to_index[(residue_index, atom_name)] for atom_name in chi_atoms]
                    chi_indices[residue_index, chi, :len(chi_atom_indices)] = torch.tensor(chi_atom_indices)
                    chi_angles[start:end, chi] = calc_dihedral(*coords[chi_atom_indices])
        start = end
    return chi_angles, chi_indices


def get_rotation(v1, v2, v3, v4, dihedral_angle):
    ab, cb, db = v2 - v1, v3 - v2, v4 - v3
    u, v = torch.cross(ab, cb), torch.cross(db, cb)
    w = torch.cross(u, v)
    rotation = Rotation.from_rotvec(dihedral_angle.cpu() * w.cpu())
    return torch.Tensor(rotation.as_matrix())