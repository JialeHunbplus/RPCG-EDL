import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import os
from paths import DATA_ROOT

# ================= 配置区 =================
# 每次跑不同数据集时，只需修改 DATA_PATH
# 建议先跑训练集，再跑测试集，它们会自动存入同一个字典中
DATA_PATH = DATA_ROOT / "kcatkm" / "train" / "Kcatkm_total.csv"
# 独立测试集特征也可按同一方法增量生成。
# DATA_PATH = DATA_ROOT / "independent_test" / "Indepent_data_clean_806.csv"

SAVE_PATH = DATA_ROOT / "kcatkm" / "substrate_pointcloud.npy"
MAX_ATOMS = 100 
# ==========================================

def get_3d_point_cloud(smiles):
    """提取 4 通道的基础几何点云 (X, Y, Z, 原子序数)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return np.zeros((4, MAX_ATOMS), dtype=np.float32)
    mol = Chem.AddHs(mol)
    
    # 尝试生成 3D 构象
    res = AllChem.EmbedMolecule(mol, randomSeed=2026)
    if res == -1:
        params = AllChem.ETKDGv3()
        params.useRandomCoords = True
        res = AllChem.EmbedMolecule(mol, params)
        if res == -1: return np.zeros((4, MAX_ATOMS), dtype=np.float32)

    try: AllChem.UFFOptimizeMolecule(mol)
    except: pass

    try: conf = mol.GetConformer()
    except: return np.zeros((4, MAX_ATOMS), dtype=np.float32)

    num_atoms = mol.GetNumAtoms()
    # 【核心修改】：张量维度还原为 (4, MAX_ATOMS)
    pc_features = np.zeros((4, MAX_ATOMS), dtype=np.float32)

    for i in range(min(num_atoms, MAX_ATOMS)):
        pos = conf.GetAtomPosition(i)
        atom = mol.GetAtomWithIdx(i)
        
        # 1. X 坐标
        pc_features[0, i] = pos.x
        # 2. Y 坐标
        pc_features[1, i] = pos.y
        # 3. Z 坐标
        pc_features[2, i] = pos.z
        # 4. 原子序数 (基础属性)
        pc_features[3, i] = atom.GetAtomicNum()
        
    valid_atoms = min(num_atoms, MAX_ATOMS)
    if valid_atoms > 0:
        # 对前3行（即 XYZ 坐标）进行中心化
        centroid = np.mean(pc_features[:3, :valid_atoms], axis=1, keepdims=True)
        pc_features[:3, :valid_atoms] -= centroid
        
    return pc_features

def main():
    print(f"🚀 当前处理数据集: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    # 自动识别列名（兼容 Protein/Sequence 或 SMILES/smiles）
    smiles_col = 'SMILES' if 'SMILES' in df.columns else 'smiles'
    unique_smiles = df[smiles_col].dropna().unique()
    
    # 增量加载逻辑
    if os.path.exists(SAVE_PATH):
        pc_dict = np.load(SAVE_PATH, allow_pickle=True).item()
        print(f"📦 发现现有字典，已载入 {len(pc_dict)} 个分子的特征。")
    else:
        pc_dict = {}
        print("🆕 未发现现有字典，将创建新字典。")
        
    new_count = 0
    for smiles in tqdm(unique_smiles, desc="提取进度"):
        if smiles not in pc_dict:
            pc_dict[smiles] = get_3d_point_cloud(smiles)
            new_count += 1
            
    # 保存更新后的字典
    np.save(SAVE_PATH, pc_dict)
    print(f"✅ 处理完成！新增: {new_count} 个，总计: {len(pc_dict)} 个。保存至: {SAVE_PATH}")

if __name__ == "__main__":
    main()
