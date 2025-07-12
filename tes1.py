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
    
    # Get per-atom EState
    try:
        estates = Chem.EState.EStateIndices(mol)
    except:
        estates = [0.0] * mol.GetNumAtoms()

    # Get per-atom TPSA and LogP/MR contributions
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

    # Convert to tensor of shape [2, num_edges]
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

        # Undirected: add both directions
        edge_index.append((i, j))
        edge_attr.append(bond_feats)

        edge_index.append((j, i))
        edge_attr.append(bond_feats)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return edge_index, edge_attr
    


class TCNBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=2, num_layers=2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(weight_norm(nn.Conv1d(
                input_dim, hidden_dim, kernel_size,
                padding=(kernel_size - 1) * dilation,
                dilation=dilation
            )))
            layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):  # x: [batch, input_dim, time]
        return self.network(x)




class GINEModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, edge_dim, out_dim, num_layers):
        super().__init__()
        self.input_projection = (
            nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        )

        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers.append(
            GINEConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim)
                ),
                edge_dim = edge_dim
            )
        )

        for _ in range(num_layers - 1):
            self.layers.append(
                GINEConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden_dim)
                    ),
                    edge_dim = edge_dim
                )
            )

        self.dropout = nn.Dropout(0.1)
        self.tcn = TCNBlock(input_dim=hidden_dim, hidden_dim=hidden_dim)
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

        graph_seq = torch.stack(layer_outputs, dim=1)  # [batch, num_layers+1, hidden_dim]
        graph_seq = graph_seq.permute(0, 2, 1)         # [batch, hidden_dim, time_steps]

        tcn_out = self.tcn(graph_seq)                  # [batch, hidden_dim, time_steps]
        final_rep = tcn_out[:, :, -1]                  # last time step

        return self.final_linear(final_rep)


    
