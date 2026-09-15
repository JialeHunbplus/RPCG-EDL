import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from transformers import T5Tokenizer, T5EncoderModel
from tqdm.auto import tqdm
from paths import DATA_ROOT, MOLT5_PATH

# ================= 路径配置 =================
IND_DATA_PATH = DATA_ROOT / "independent_test" / "Indepent_data_clean_806.csv"
DICT_PATH = DATA_ROOT / "kcatkm" / "substrate_features.npy"
# ============================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = T5Tokenizer.from_pretrained(str(MOLT5_PATH), do_lower_case=False)
model = T5EncoderModel.from_pretrained(str(MOLT5_PATH)).to(device)
if torch.cuda.is_available():
    model = model.half()
model.eval()

def get_maccs_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: mol = Chem.MolFromSmiles('')
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(list(map(int, fp.ToBitString())), dtype=np.float32)

print("正在加载域外测试集数据...")
df = pd.read_csv(IND_DATA_PATH)
unique_smiles = df['SMILES'].dropna().unique()

print("正在加载已有特征字典...")
substrate_features = np.load(DICT_PATH, allow_pickle=True).item()
initial_len = len(substrate_features)

print("开始提取域外测试集的新底物特征...")
with torch.no_grad():
    for smiles in tqdm(unique_smiles):
        if smiles in substrate_features:
            continue # 已存在的直接跳过，节省时间
        
        # 1. 提取 MACCS
        maccs_fp = get_maccs_fp(smiles)
        
        # 2. 提取 MolT5
        inputs = tokenizer(smiles, return_tensors="pt").to(device)
        outputs = model(input_ids=inputs['input_ids'])
        molt5_embed = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        
        # 追加到字典
        substrate_features[smiles] = {
            'molt5': molt5_embed,
            'maccs': maccs_fp
        }

np.save(DICT_PATH, substrate_features)
print(f"追加完成！新增了 {len(substrate_features) - initial_len} 个底物特征。")
print(f"当前字典总容量: {len(substrate_features)}")
