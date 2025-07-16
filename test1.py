import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import polars as pl

import gc
import pickle

import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

import lightgbm as lgb

from sklearn.model_selection import KFold

import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops

class CFG:
    TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    SEED = 42
    FOLDS = 5
    PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/'


TARGETS2 = ['FFV', 'Tc', 'Density', 'Rg']

train = pd.read_csv(CFG.PATH + 'train.csv')
test = pd.read_csv(CFG.PATH + 'test.csv')

def make_smile_canonical(smile): 
    try:
        mol = Chem.MolFromSmiles(smile)
        canon_smile = Chem.MolToSmiles(mol, canonical=True)
        return canon_smile
    except:
        return np.nan

useless_cols = [    
    # Nan data
    'BCUT2D_MWHI',
    'BCUT2D_MWLOW',
    'BCUT2D_CHGHI',
    'BCUT2D_CHGLO',
    'BCUT2D_LOGPHI',
    'BCUT2D_LOGPLOW',
    'BCUT2D_MRHI',
    'BCUT2D_MRLOW',

    # Constant data
    'NumRadicalElectrons',
    'SMR_VSA8',
    'SlogP_VSA9',
    'fr_barbitur',
    'fr_benzodiazepine',
    'fr_dihydropyridine',
    'fr_epoxide',
    'fr_isothiocyan',
    'fr_lactam',
    'fr_nitroso',
    'fr_prisulfonamd',
    'fr_thiocyan',

    # High correlated data >0.95
    'MaxEStateIndex',
    'HeavyAtomMolWt',
    'ExactMolWt',
    'NumValenceElectrons',
    'Chi0',
    'Chi0n',
    'Chi0v',
    'Chi1',
    'Chi1n',
    'Chi1v',
    'Chi2n',
    'Kappa1',
    'LabuteASA',
    'HeavyAtomCount',
    'MolMR',
    'Chi3n',
    'BertzCT',
    'Chi2v',
    'Chi4n',
    'HallKierAlpha',
    'Chi3v',
    'Chi4v',
    'MinAbsPartialCharge',
    'MinPartialCharge',
    'MaxAbsPartialCharge',
    'FpDensityMorgan2',
    'FpDensityMorgan3',
    'Phi',
    'Kappa3',
    'fr_nitrile',
    'SlogP_VSA6',
    'NumAromaticCarbocycles',
    'NumAromaticRings',
    'fr_benzene',
    'VSA_EState6',
    'NOCount',
    'fr_C_O',
    'fr_C_O_noCOO',
    'NumHDonors',
    'fr_amide',
    'fr_Nhpyrrole',
    'fr_phenol',
    'fr_phenol_noOrthoHbond',
    'fr_COO2',
    'fr_halogen',
    'fr_diazo',
    'fr_nitro_arom',
    'fr_phos_ester'
]

def compute_all_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None] * len(desc_names)
    return [desc[1](mol) for desc in Descriptors.descList if desc[0] not in useless_cols]

def compute_graph_features(smiles, graph_feats):
    mol = Chem.MolFromSmiles(smiles)
    adj = rdmolops.GetAdjacencyMatrix(mol)
    G = nx.from_numpy_array(adj)

    graph_feats['graph_diameter'].append(nx.diameter(G) if nx.is_connected(G) else 0)
    graph_feats['avg_shortest_path'].append(nx.average_shortest_path_length(G) if nx.is_connected(G) else 0)
    graph_feats['num_cycles'].append(len(list(nx.cycle_basis(G))))

def preprocessing(df):
    desc_names = [desc[0] for desc in Descriptors.descList if desc[0] not in useless_cols]
    print(desc_names)
    descriptors = [compute_all_descriptors(smi) for smi in df['SMILES'].to_list()]

    graph_feats = {'graph_diameter': [], 'avg_shortest_path': [], 'num_cycles': []}
    for smile in df['SMILES']:
         compute_graph_features(smile, graph_feats)
        
    result = pd.concat(
        [
            pd.DataFrame(descriptors, columns=desc_names),
            pd.DataFrame(graph_feats)
        ],
        axis=1
    )

    result = result.replace([-np.inf, np.inf], np.nan)
    return result

train = pd.concat([train, preprocessing(train)], axis=1)
test = pd.concat([test, preprocessing(test)], axis=1)

all_features = train.columns[7:].tolist()
features = {}
for target in CFG.TARGETS:
    const_descs = []
    for col in train.columns.drop(CFG.TARGETS):
        if train[train[target].notnull()][col].nunique() == 1:
            const_descs.append(col)
    features[target] = [f for f in all_features if f not in const_descs]

train['SMILES'] = train['SMILES'].apply(lambda s: make_smile_canonical(s))
test['SMILES'] = test['SMILES'].apply(lambda s: make_smile_canonical(s))

ATOM_TYPES = [
    'C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P',
    'H', 'B', 'Si', 'Se', 'As', 'other'
]
ATOM_TYPES_data = {'C' :[], 'N': [], 'O': [], 'S': [], 'F': [],
                  'Cl': [], 'Br': [], 'I': [], 'P':[], 'H': [],
                  'B': [], 'Si': [], 'Se': [], 'As': [], 'other': []}


Relevant_Data = {'Aromatic': [], 'In Ring': [], 'Formal Charge': [], 'sp': [], 'sp1': [], 'sp2': [], 'sp3':[],
                 'sp3d': [], 'sp3d2': [], 'Number of Hs': [], 'Donor': [], 'Acceptor': [], 'Atomic Mass': [], 'Electronegativity': [],
                'TPSA': [], 'LogP': [], 'Gasteiger Charge': [], 'EState Val:' []}

HYBRIDIZATION_TYPES = [
    rdchem.HybridizationType.SP, 
    rdchem.HybridizationType.SP2, 
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]

hybrid = atom.GetHybridization()
one_hot = [int(hybrid == htype) for htype in HYBRIDIZATION_TYPES]


ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'H', 'B', 'Si', 'Se', 'As', 'other']
HYBRIDIZATION_TYPES = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]

def preprocessing_node_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    AllChem.ComputeGasteigerCharges(mol)
    
   
    try:
        estates = Chem.EState.EStateIndices(mol)
    except:
        estates = [0.0] * mol.GetNumAtoms()

    
    tpsa_contribs = rdMolDescriptors._CalcTPSAContribs(mol)
    crippen_contribs = rdMolDescriptors._CalcCrippenContribs(mol)

    atom_features = []

    for idx, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        symbol = symbol if symbol in ATOM_TYPES else 'other'

        atom_type_feats = [int(symbol == t) for t in ATOM_TYPES]
        hybrid = atom.GetHybridization()
        hybrid_feats = [int(hybrid == htype) for htype in HYBRIDIZATION_TYPES]

        feat = atom_type_feats + hybrid_feats + [
            atom.GetFormalCharge(),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
            atom.GetTotalNumHs(),
            int(atom.GetTotalNumHs() > 0 and atom.GetAtomicNum() in [7, 8]),  # Donor (rough)
            int(atom.GetAtomicNum() in [7, 8]),  # Acceptor (simplified)
            atom.GetMass(),
            Chem.GetPeriodicTable().GetElectronegativity(atom.GetAtomicNum()),
            tpsa_contribs[idx],
            crippen_contribs[idx][0],  # LogP
            float(atom.GetProp('_GasteigerCharge')) if atom.HasProp('_GasteigerCharge') else 0.0,
            estates[idx]
        ]

        atom_features.append(feat)

    return atom_features


def compute_edge_connectivity(smiles):
    mol = Chem.MolFromSmiles(smiles)
    edge_index = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append((i, j))
        edge_index.append((j, i))  # undirected graph

    #shape [2, num_edges]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index
    

def compute_edge_features(smiles):
    mol = Chem.MolFromSmiles(smiles)

    edge_index = []
    edge_attr = []

    bond_type_to_idx = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3
    }

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # One-hot bond type
        bond_type = bond_type_to_idx.get(bond.GetBondType(), 0)
        bond_type_onehot = [int(bond_type == k) for k in range(4)]

        bond_feats = bond_type_onehot + [
            int(bond.GetIsAromatic()),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing())
        ]

        # Undirected
        edge_index.append((i, j))
        edge_attr.append(bond_feats)

        edge_index.append((j, i))
        edge_attr.append(bond_feats)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return edge_index, edge_attr


class AugmentedSMILESDataset(Dataset):
    def __init__(self, df, target, num_aug=3, transform=None, pre_transform=None):
        self.df = df.reset_index(drop=True)  
        self.target = target
        self.num_aug = num_aug
        super().__init__(root=None, transform=transform, pre_transform=pre_transform)

    def len(self):
        return len(self.df)

    def get(self, idx):
        smiles = self.df.loc[idx, 'SMILES']
        y_val = self.df.loc[idx, self.target]

        
        if self.num_aug > 0 and random.random() < 0.5:
            smiles = random.choice(randomize_smiles(smiles, self.num_aug))

        data = smiles_to_graph_data(smiles, {'y': y_val})

        
        if data is None:
            return Data(
                x=torch.zeros((1, 1)),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                y=torch.tensor([y_val], dtype=torch.float)
            )
        return data

def randomize_smiles(smiles, num_aug=1):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    return [Chem.MolToSmiles(mol, doRandom=True) for _ in range(num_aug)]

def train_loop(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze()
        loss = criterion(out, batch.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)




def eval_loop(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds = []
    trues = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze()
            loss = criterion(out, batch.y.squeeze())
            total_loss += loss.item() * batch.num_graphs
            preds.append(out.cpu())
            trues.append(batch.y.cpu().squeeze())
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()
    return total_loss / len(loader.dataset), preds, trues
    



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool

class GINEModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, edge_dim, out_dim, num_layers, num_heads=4):
        super().__init__()
        self.input_projection = (
            nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        )

        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        
        for _ in range(num_layers):
            self.layers.append(
                GINEConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden_dim)
                    ),
                    edge_dim=edge_dim
                )
            )

        self.dropout = nn.Dropout(0.5)

        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            bidirectional=True,
            batch_first=True
        )

      
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        self.final_linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.input_projection(x)

        layer_outputs = [global_mean_pool(x, batch)]
        for conv in self.layers:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = self.dropout(x)
            pooled = global_mean_pool(x, batch)
            layer_outputs.append(pooled)

        # [batch, seq_len, hidden_dim]
        graph_seq = torch.stack(layer_outputs, dim=1)

        
        lstm_out, _ = self.lstm(graph_seq)  # [batch, seq_len, hidden_dim]

        
        attn_out, _ = self.multihead_attn(lstm_out, lstm_out, lstm_out)  # [batch, seq_len, hidden_dim]

        attn_output = attn_out.mean(dim=1)  # [batch, hidden_dim]

        return self.final_linear(attn_output)








    
