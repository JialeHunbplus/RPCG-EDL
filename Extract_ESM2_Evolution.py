import torch
import pandas as pd
import numpy as np
from transformers import EsmTokenizer, EsmForMaskedLM
from tqdm.auto import tqdm
import os
from paths import DATA_ROOT, ESM2_PATH

# ================= 配置区域 =================
MODEL_NAME = ESM2_PATH
DATA_PATH = DATA_ROOT / "independent_test" / "Indepent_data_clean_806.csv"
OUTPUT_DICT_PATH = DATA_ROOT / "kcatkm" / "protein_esm2_evo.npy"
# ============================================

print(f"Loading model: {MODEL_NAME} ...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = EsmTokenizer.from_pretrained(str(MODEL_NAME))
model = EsmForMaskedLM.from_pretrained(str(MODEL_NAME)).to(device)
if torch.cuda.is_available():
    model = model.half()
model.eval()

standard_aas = list("ACDEFGHIKLMNPQRSTVWY")
aa_indices = [tokenizer.convert_tokens_to_ids(aa) for aa in standard_aas]

print(f"Reading data from {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
# 核心修改：已自动适配你的列名 'Protein'
unique_seqs = df['Protein'].dropna().unique()

esm2_features = {}
if os.path.exists(OUTPUT_DICT_PATH):
    esm2_features = np.load(OUTPUT_DICT_PATH, allow_pickle=True).item()
    print(f"Found existing dict: {len(esm2_features)} records.")

print("Extracting ESM-2 features...")
with torch.no_grad():
    for seq in tqdm(unique_seqs):
        if seq in esm2_features:
            continue
            
        truncated_seq = seq[:1022] 
        inputs = tokenizer(truncated_seq, return_tensors="pt", add_special_tokens=True).to(device)
        
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(0)[1:-1, :] 
        
        aa_logits = logits[:, aa_indices] 
        evo_probs = torch.softmax(aa_logits, dim=-1).cpu().numpy() 
        
        esm2_features[seq] = evo_probs

np.save(OUTPUT_DICT_PATH, esm2_features)
print(f"Done. Saved {len(esm2_features)} records to {OUTPUT_DICT_PATH}")
