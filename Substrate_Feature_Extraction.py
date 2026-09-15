import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from transformers import T5Tokenizer, T5EncoderModel
from tqdm.auto import tqdm
import os
from paths import DATA_ROOT, MOLT5_PATH

# 配置路径
DATA_PATH = DATA_ROOT / "kcatkm" / "train" / "Kcatkm_total.csv"
SAVE_PATH = DATA_ROOT / "kcatkm" / "substrate_features.npy"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("正在加载 MolT5 模型...")
tokenizer = T5Tokenizer.from_pretrained(str(MOLT5_PATH), do_lower_case=False)
model = T5EncoderModel.from_pretrained(str(MOLT5_PATH)).to(device)
model.eval()
if torch.cuda.is_available():
    model = model.half() # 开启半精度，RTX 5090 完全支持且提速

def get_maccs_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: mol = Chem.MolFromSmiles('') # 容错
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(list(map(int, fp.ToBitString())), dtype=np.float32)

print("正在读取数据...")
df = pd.read_csv(DATA_PATH)
unique_smiles = df['SMILES'].dropna().unique()

substrate_features = {}

print("开始提取 MolT5 与 MACCS 特征...")
with torch.no_grad():
    for smiles in tqdm(unique_smiles):
        # 1. 提取 MACCS
        maccs_fp = get_maccs_fp(smiles)
        
        # 2. 提取 MolT5
        inputs = tokenizer(smiles, return_tensors="pt").to(device)
        outputs = model(input_ids=inputs['input_ids'])
        # 取 last_hidden_state 的平均池化作为全局语义向量
        molt5_embed = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        
        substrate_features[smiles] = {
            'molt5': molt5_embed,
            'maccs': maccs_fp
        }

np.save(SAVE_PATH, substrate_features)
print(f"特征已成功保存至 {SAVE_PATH}！")
