import pandas as pd
import torch.utils.data as data
import torch
import numpy as np
from fragment_graph import build_brics_fragment_graph
from utils import integer_label_protein
from Prot_fasta_Feature_Extraction import embed_dataset
from paths import DATA_ROOT


class DTIDataset(data.Dataset):
    def __init__(self, list_IDs, df, cfg):
        self.list_IDs = list_IDs
        self.df = df
        self.cfg = cfg
        
        # 1. 加载底物语义与结构特征字典 (MolT5 + MACCS)
        feature_dict_path = DATA_ROOT / "kcatkm" / "substrate_features.npy"
        try:
            self.substrate_features = np.load(feature_dict_path, allow_pickle=True).item()
            print(f"成功加载底物语义特征字典，包含 {len(self.substrate_features)} 个分子。")
        except FileNotFoundError:
            raise FileNotFoundError(f"未找到特征字典文件：{feature_dict_path}，请先运行预提取脚本！")
            
        # 2. 【新增】加载底物 3D 点云空间特征字典 (Point Cloud)
        pc_dict_path = DATA_ROOT / "kcatkm" / "substrate_pointcloud.npy"
        try:
            self.pc_dict = np.load(pc_dict_path, allow_pickle=True).item()
            print(f"成功加载底物 3D 点云特征字典，包含 {len(self.pc_dict)} 个分子。")
        except FileNotFoundError:
            print(f"Warning: 未找到 {pc_dict_path}。如果你打算使用空间特征，请先运行 Extract_PointCloud_Features.py！")
            self.pc_dict = {}

        self.fragment_graph_cache = {}

        # 3. 加载蛋白质进化特征字典 (ESM-2)
        esm2_dict_path = DATA_ROOT / "kcatkm" / "protein_esm2_evo.npy"
        try:
            self.esm2_features = np.load(esm2_dict_path, allow_pickle=True).item()
            print(f"成功加载 ESM-2 进化特征字典，包含 {len(self.esm2_features)} 个蛋白质。")
        except FileNotFoundError:
            raise FileNotFoundError(f"未找到 ESM-2 字典：{esm2_dict_path}，请先运行 Extract_ESM2_Evolution.py！")

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        
        # =========================================================
        # 底物特征处理 (三塔融合：MolT5 + MACCS + PointCloud)
        # =========================================================
        smiles = self.df.iloc[index]['SMILES']
        
        # 提取 1D 语义与 2D 结构
        if smiles in self.substrate_features:
            feat = self.substrate_features[smiles]
            molt5_embed = torch.tensor(feat['molt5'], dtype=torch.float32)
            maccs_fp = torch.tensor(feat['maccs'], dtype=torch.float32)
        else:
            molt5_embed = torch.zeros(768, dtype=torch.float32)
            maccs_fp = torch.zeros(167, dtype=torch.float32)
            
        # 提取 3D 点云空间特征 [4, 100]
        # 如果字典中没有该分子，返回全 0 占位
        pc_feat_np = self.pc_dict.get(smiles, np.zeros((4, 100)))
        pc_feature = torch.tensor(pc_feat_np, dtype=torch.float32)
            
        if smiles not in self.fragment_graph_cache:
            self.fragment_graph_cache[smiles] = build_brics_fragment_graph(smiles)
        frag_graph = self.fragment_graph_cache[smiles]

        # v_d 现在是包含四个模态特征的元组：MolT5、MACCS、3D点云、BRICS片段图
        v_d = (molt5_embed, maccs_fp, pc_feature, frag_graph)

        # =========================================================
        # 蛋白质特征处理 (ProtT5 + ESM-2 融合)
        # =========================================================
        seq = self.df.iloc[index]['Protein']
        
        # 提取 ProtT5 结构语义
        if self.cfg['protein_encode']['input'] == 'prottrans':
            v_p = embed_dataset(seq)
            pad_width = ((0, 1200-len(v_p)), (0, 0))
            v_p = np.pad(v_p, pad_width, mode='constant', constant_values=0)
        elif self.cfg['protein_encode']['input'] == 'integer':
            v_p = integer_label_protein(seq)
            
        v_p = torch.tensor(v_p, dtype=torch.float32)
        
        # 提取并对齐 ESM-2 进化保守性概率 [Seq_Len, 20]
        if seq in self.esm2_features:
            esm2_feat = torch.tensor(self.esm2_features[seq], dtype=torch.float32)
        else:
            esm2_feat = torch.zeros((v_p.shape[0], 20), dtype=torch.float32)
            
        L_prot = v_p.shape[0]
        L_esm = esm2_feat.shape[0]
        
        # 动态长度对齐
        if L_esm > L_prot:
            esm2_feat = esm2_feat[:L_prot, :]
        elif L_esm < L_prot:
            pad_len = L_prot - L_esm
            padding = torch.zeros((pad_len, 20), dtype=torch.float32)
            esm2_feat = torch.cat([esm2_feat, padding], dim=0)
            
        v_p = torch.cat([v_p, esm2_feat], dim=-1)

        # =========================================================
        # 标签与权重处理
        # =========================================================
        y = np.log10(self.df.iloc[index]["Y"])
        w = self.df.iloc[index]["w"]
        
        return v_d, v_p, y, w
