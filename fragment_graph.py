import numpy as np
import torch
import dgl
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import BRICS, MACCSkeys

RDLogger.DisableLog('rdApp.*')

def _fragment_maccs_feature(mol):
    arr = np.zeros((167,), dtype=np.float32)
    if mol is None:
        return arr
    try:
        fp = MACCSkeys.GenMACCSKeys(mol)
        DataStructs.ConvertToNumpyArray(fp, arr)
    except Exception:
        pass
    return arr

def build_brics_fragment_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        graph = dgl.graph(([], []), num_nodes=1)
        graph.ndata["h"] = torch.zeros((1, 167), dtype=torch.float32)
        return dgl.add_self_loop(graph)

    brics_bonds = list(BRICS.FindBRICSBonds(mol))
    if not brics_bonds:
        graph = dgl.graph(([], []), num_nodes=1)
        graph.ndata["h"] = torch.tensor(
            _fragment_maccs_feature(mol).reshape(1, -1),
            dtype=torch.float32,
        )
        return dgl.add_self_loop(graph)

    cut_mol = Chem.RWMol(mol)
    cut_pairs = [(int(pair[0]), int(pair[1])) for pair, _ in brics_bonds]
    for a_idx, b_idx in cut_pairs:
        if cut_mol.GetBondBetweenAtoms(a_idx, b_idx) is not None:
            cut_mol.RemoveBond(a_idx, b_idx)
    cut_mol = cut_mol.GetMol()

    atom_frags = Chem.GetMolFrags(cut_mol, asMols=False, sanitizeFrags=False)
    frag_mols = []
    for atoms in atom_frags:
        try:
            frag_smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=list(atoms), canonical=True)
            frag_mols.append(Chem.MolFromSmiles(frag_smiles))
        except Exception:
            frag_mols.append(None)

    atom_to_frag = {}
    for frag_idx, atoms in enumerate(atom_frags):
        for atom_idx in atoms:
            atom_to_frag[int(atom_idx)] = frag_idx

    features = [_fragment_maccs_feature(frag_mol) for frag_mol in frag_mols]
    if not features:
        features = [_fragment_maccs_feature(mol)]

    src, dst = [], []
    for a_idx, b_idx in cut_pairs:
        fa = atom_to_frag.get(a_idx)
        fb = atom_to_frag.get(b_idx)
        if fa is not None and fb is not None and fa != fb:
            src.extend([fa, fb])
            dst.extend([fb, fa])

    graph = dgl.graph((src, dst), num_nodes=len(features))
    graph.ndata["h"] = torch.tensor(np.stack(features), dtype=torch.float32)
    return dgl.add_self_loop(graph)
