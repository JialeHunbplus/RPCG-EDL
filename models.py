import torch.nn as nn
import torch.nn.functional as F
import torch
import dgl
from dgl.nn import GraphConv
from ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm

# ==========================================
# 1. 底物端模块：蛋白条件化四模态门控 (MolT5 + MACCS + 点云 + BRICS)
# ==========================================

class SubstratePointNet(nn.Module):
    def __init__(self, in_channels=4, out_dim=128):
        super(SubstratePointNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, out_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=False)[0] 
        return x


class FragmentGraphEncoder(nn.Module):
    def __init__(self, in_dim=167, hidden_dim=256, out_dim=256, dropout=0.2):
        super(FragmentGraphEncoder, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim, allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_dim, out_dim, allow_zero_in_degree=True)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, graph):
        h = graph.ndata["h"].to(torch.float32)
        h = self.conv1(graph, h)
        h = F.relu(self.norm1(h))
        h = self.dropout(h)
        h = self.conv2(graph, h)
        h = F.relu(self.norm2(h))
        graph.ndata["frag_h"] = h
        return dgl.mean_nodes(graph, "frag_h")


class MolecularSemanticEncoder(nn.Module):
    def __init__(self, molt5_dim=768, maccs_dim=167, frag_node_dim=167, hidden_dim=256, out_dim=128, dropout=0.2):
        super(MolecularSemanticEncoder, self).__init__()
        
        print("\n" + "="*45)
        print("🚀 残差蛋白条件化底物门控 (MolT5+MACCS+3D+BRICS片段图)")
        print("="*45 + "\n")
        
        self.proj_molt5 = nn.Sequential(
            nn.Linear(molt5_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.proj_maccs = nn.Sequential(
            nn.Linear(maccs_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.pc_net = SubstratePointNet(in_channels=4, out_dim=hidden_dim)
        self.frag_net = FragmentGraphEncoder(in_dim=frag_node_dim, hidden_dim=hidden_dim, out_dim=hidden_dim, dropout=dropout)

        self.protein_condition = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.base_gate_layer = nn.Linear(hidden_dim * 4, 4)
        self.condition_delta_layer = nn.Linear(hidden_dim * 5, 4)
        self.condition_scale = 0.1
        self.last_gate_info = None
        self.mixer = nn.Sequential(
            nn.Linear(hidden_dim * 4, out_dim),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, molt5_embed, maccs_fp, pc_feature, frag_graph, protein_context):
        h_mol = self.proj_molt5(molt5_embed) 
        h_mac = self.proj_maccs(maccs_fp)    
        h_pc = F.relu(self.pc_net(pc_feature))
        h_frag = self.frag_net(frag_graph)
        h_protein = self.protein_condition(protein_context)
        
        combined = torch.cat([h_mol, h_mac, h_pc, h_frag], dim=1)
        base_gate_logit = self.base_gate_layer(combined)
        base_gate = torch.sigmoid(base_gate_logit)
        condition_input = torch.cat([combined, h_protein], dim=1)
        gate_delta = self.condition_scale * torch.tanh(self.condition_delta_layer(condition_input))
        preclip_gate = base_gate + gate_delta
        gate_weights = torch.clamp(preclip_gate, 0.0, 1.0)
        self.last_gate_info = {
            "base_gate_logit": base_gate_logit.detach(),
            "base_gate": base_gate.detach(),
            "condition_delta": gate_delta.detach(),
            "preclip_gate": preclip_gate.detach(),
            "final_gate": gate_weights.detach(),
            "effective_multiplier": (1.0 + gate_weights).detach(),
        }
        
        if self.training and torch.rand(1).item() < 0.01:
            print(
                f"DEBUG 残差条件门控 -> "
                f"base: [{base_gate[0,0].item():.3f}, {base_gate[0,1].item():.3f}, {base_gate[0,2].item():.3f}, {base_gate[0,3].item():.3f}] | "
                f"delta: [{gate_delta[0,0].item():.3f}, {gate_delta[0,1].item():.3f}, {gate_delta[0,2].item():.3f}, {gate_delta[0,3].item():.3f}]"
            )

        gated_mol = h_mol * gate_weights[:, 0:1]
        gated_mac = h_mac * gate_weights[:, 1:2]
        gated_pc = h_pc * gate_weights[:, 2:3]
        gated_frag = h_frag * gate_weights[:, 3:4]

        gated_combined = torch.cat([gated_mol, gated_mac, gated_pc, gated_frag], dim=1) + combined
        v_d = self.mixer(gated_combined)
        return v_d.unsqueeze(1)


# ==========================================
# 2. 蛋白质端模块：整体 1044 维协同融合 (ProtT5 + ESM2)
# ==========================================

class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(ProteinCNN, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        v = self.embedding(v.long()) 
        v = v.transpose(2, 1)
        v = self.bn1(F.relu(self.conv1(v))) 
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))
        v = v.view(v.size(0), v.size(2), -1)
        return v

class pretrain_ProteinCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(pretrain_ProteinCNN, self).__init__()
        self.embedding = nn.Linear(1024, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        v = self.embedding(v.to(torch.float32)) 
        v = v.transpose(2, 1)
        v = self.bn1(F.relu(self.conv1(v))) 
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))
        v = v.view(v.size(0), v.size(2), -1)
        return v

class LightAttention(nn.Module):
    def __init__(self, embedding_dim=128, conv_embeddings_dim=1044, dropout=0.25, kernel_size=9):
        super(LightAttention, self).__init__()
        self.feature_convolution = nn.Conv1d(conv_embeddings_dim, conv_embeddings_dim, kernel_size, stride=1, padding=kernel_size // 2)
        self.attention_convolution = nn.Conv1d(conv_embeddings_dim, conv_embeddings_dim, kernel_size, stride=1, padding=kernel_size // 2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Sequential(
            nn.Linear(2 * conv_embeddings_dim, embedding_dim),  
            nn.Dropout(dropout), 
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x= x.to(torch.float32)
        x = x.transpose(2,1)
        o = self.feature_convolution(x)  
        o = self.dropout(o)  
        attention = self.attention_convolution(x)  
        o1 = o * self.softmax(attention)  
        o2 = o  
        o = torch.cat([o1, o2], dim=1)  
        o = o.transpose(2, 1)   
        o = self.linear(o)  
        return o


# ==========================================
# 3. 辅助预测器与大统一网络
# ==========================================

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

class RPCGEDL(nn.Module):
    def __init__(self, **config):
        super().__init__()
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        ban_heads = config["BCN"]["HEADS"]
        protein_input = config['protein_encode']['input']
        protein_encoder = config['protein_encode']['encoder']
        
        self.drug_extractor = MolecularSemanticEncoder(
            molt5_dim=config["DRUG"].get("MOLT5_DIM", 768),
            maccs_dim=config["DRUG"].get("MACCS_DIM", 167),
            frag_node_dim=config["DRUG"].get("FRAG_NODE_DIM", 167),
            out_dim=drug_embedding
        )

        prot_dim = config["PROTEIN_LA"]["conv_embeddings_dim"]
        self.prot_norm = nn.LayerNorm(prot_dim)
        if protein_input == 'integer' and protein_encoder=='cnn':
            protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
            num_filters = config["PROTEIN"]["NUM_FILTERS"]
            kernel_size = config["PROTEIN"]["KERNEL_SIZE"]
            self.protein_extractor = ProteinCNN(protein_emb_dim, num_filters, kernel_size, protein_padding)
        elif protein_input == 'prottrans' and protein_encoder=='cnn':
            protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
            num_filters = config["PROTEIN"]["NUM_FILTERS"]
            kernel_size = config["PROTEIN"]["KERNEL_SIZE"]
            self.protein_extractor = pretrain_ProteinCNN(protein_emb_dim, num_filters, kernel_size, protein_padding)
        elif protein_input == 'prottrans' and protein_encoder == 'LA':
            protein_emb_dim = config["PROTEIN_LA"]["EMBEDDING_DIM"]
            conv_embeddings_dim = config["PROTEIN_LA"]["conv_embeddings_dim"] # 接收1044维
            kernel_size = config["PROTEIN_LA"]["KERNEL_SIZE"]
            dropout = config["PROTEIN_LA"]["dropout"]
            self.protein_extractor = LightAttention(protein_emb_dim, conv_embeddings_dim, kernel_size=kernel_size, dropout=dropout)

        num_filters_last = config["PROTEIN"]["NUM_FILTERS"][-1] if protein_encoder=='cnn' else config["PROTEIN_LA"]["EMBEDDING_DIM"]
        self.bcn = weight_norm(
            BANLayer(v_dim=drug_hidden_feats[-1], q_dim=num_filters_last, h_dim=mlp_in_dim, h_out=ban_heads),
            name='h_mat', dim=None)
        
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, v_d, v_p, mode="train"):
        molt5_embed, maccs_fp, pc_feature, frag_graph = v_d
        if isinstance(v_p, tuple):
            v_p = v_p[0]
        v_p = self.prot_norm(v_p)
        v_p = self.protein_extractor(v_p) 
        protein_context = v_p.mean(dim=1)
        v_d = self.drug_extractor(molt5_embed, maccs_fp, pc_feature, frag_graph, protein_context)
        
        f, att = self.bcn(v_d, v_p)
        score = self.mlp_classifier(f)
        
        min_val = 1e-6
        means, loglambdas, logalphas, logbetas = torch.split(score, score.shape[1] // 4, dim=1)
        lambdas = torch.nn.Softplus()(loglambdas) + min_val
        alphas = torch.nn.Softplus()(logalphas) + min_val + 1  
        betas = torch.nn.Softplus()(logbetas) + min_val

        score = torch.stack((means, lambdas, alphas, betas), dim=2).view(score.size())
        
        if mode == "train":
            return v_d, v_p, f, score
        elif mode == "eval":
            return v_d, v_p, score, att
