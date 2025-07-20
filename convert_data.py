import os
import argparse
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser


atom_types = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I']
atom2id = {a: i for i, a in enumerate(atom_types)}

def one_hot(val, length):
    vec = [0] * length
    if val < length:
        vec[val] = 1
    return vec

def get_atom_feature_rdkit(atom):
    return one_hot(atom2id.get(atom.GetSymbol(), len(atom_types)), len(atom_types)+1)

def get_atom_feature_biopython(element):
    return one_hot(atom2id.get(element, len(atom_types)), len(atom_types)+1)

def get_ligand_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    coords = mol.GetConformer().GetPositions()
    feats = [get_atom_feature_rdkit(atom) for atom in mol.GetAtoms()]
    edges = [[], []]
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges[0] += [i, j]
        edges[1] += [j, i]
    return torch.tensor(feats, dtype=torch.float), torch.tensor(edges, dtype=torch.long), coords

def get_sequence(pdb_file, max_len=1000):
    aa_dict = {'ALA':1,'CYS':2,'ASP':3,'GLU':4,'PHE':5,'GLY':6,'HIS':7,'ILE':8,'LYS':9,
               'LEU':10,'MET':11,'ASN':12,'PRO':13,'GLN':14,'ARG':15,'SER':16,'THR':17,'VAL':18,'TRP':19,'TYR':20}
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    residues = [res for res in structure.get_residues() if res.id[0] == " "]
    seq = [aa_dict.get(res.resname, 0) for res in residues[:max_len]]
    return torch.tensor(seq + [0]*(max_len - len(seq)), dtype=torch.long)

def get_pocket_coords(pdb_file, ligand_coords, cutoff=6.0):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    pocket = []
    for atom in structure.get_atoms():
        pos = atom.coord
        if np.min(np.linalg.norm(ligand_coords - pos, axis=1)) < cutoff:
            symbol = atom.element.strip()
            feat = get_atom_feature_biopython(symbol)
            pocket.append((pos, feat))
    return pocket

def build_pd_graph(ligand_coords, ligand_feats, pocket_coords_feats):
    all_feats = ligand_feats + [f for _, f in pocket_coords_feats]
    all_coords = np.vstack([ligand_coords, np.array([c for c, _ in pocket_coords_feats])])
    edge_index = [[], []]
    n_lig = len(ligand_coords)

    for i in range(n_lig):
        for j in range(n_lig):
            if i != j and np.linalg.norm(ligand_coords[i] - ligand_coords[j]) < 1.8:
                edge_index[0].append(i)
                edge_index[1].append(j)

    for i, lig_c in enumerate(ligand_coords):
        for j, (pock_c, _) in enumerate(pocket_coords_feats):
            if np.linalg.norm(lig_c - pock_c) < 6.0:
                pi = n_lig + j
                edge_index[0] += [i, pi]
                edge_index[1] += [pi, i]

    return torch.tensor(all_feats, dtype=torch.float), torch.tensor(edge_index, dtype=torch.long)

def process_complex(entry_dir):
    pdb_file = os.path.join(entry_dir, "protein.pdb")
    smiles_file = os.path.join(entry_dir, "ligand.smi")
    affinity_file = os.path.join(entry_dir, "affinity.txt")

    with open(smiles_file) as f:
        smiles = f.readline().strip()
    with open(affinity_file) as f:
        affinity = float(f.readline().strip())

    lig_feats, lig_edges, lig_coords = get_ligand_graph(smiles)
    seq = get_sequence(pdb_file)
    pocket_atoms = get_pocket_coords(pdb_file, lig_coords)
    pd_feats, pd_edges = build_pd_graph(lig_coords, lig_feats.tolist(), pocket_atoms)

    return {
        "protein": seq,
        "drug_graph": {"node_features": lig_feats, "edge_index": lig_edges},
        "pd_graph": {"node_features": pd_feats, "edge_index": pd_edges},
        "affinity": torch.tensor(affinity, dtype=torch.float)
    }

def main(args):
    os.makedirs(args.output, exist_ok=True)
    dataset = []
    for name in os.listdir(args.input_pdb):
        entry_path = os.path.join(args.input_pdb, name)
        if not os.path.isdir(entry_path):
            continue
        try:
            data = process_complex(entry_path)
            dataset.append(data)
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
    torch.save(dataset, os.path.join(args.output, "processed_data.pt"))
    print(f"Converted {len(dataset)} samples to {args.output}/processed_data.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pdb", type=str, required=True, help="Path to raw PDBbind-like data")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    main(args)
