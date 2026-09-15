# RPCG-EDL

RPCG-EDL predicts enzyme catalytic efficiency on the `log10(kcat/Km)` scale and produces sample-level uncertainty estimates. The substrate encoder combines MolT5 embeddings, MACCS fingerprints, 3D point clouds, and BRICS fragment graphs. The protein encoder combines ProtT5 residue embeddings with ESM-2 amino-acid probability features. Light Attention, protein-conditioned residual gating, bilinear attention, and an evidential regression head connect these representations.

## Source files

`models.py`, `ban.py`, and `fragment_graph.py` implement the network and molecular graph branch. `dataloader.py`, `trainer.py`, `utils.py`, and `configs.py` provide data loading, optimization, and configuration. `main.py` is the training entry point, and `independent_test_attention.py` runs evaluation on the independent dataset. The feature extraction scripts generate or extend the cached model inputs. `paths.py` defines locations for data and local pretrained models.

## Data and pretrained models

The default data directory is `../2/datasets/` in the local three-folder layout. When using this source directory on its own, set `RPCG_DATA_ROOT` to a directory containing `kcatkm/` and `independent_test/`. The loader expects these cached inputs in `kcatkm/`: `substrate_features.npy`, `substrate_pointcloud.npy`, and `protein_esm2_evo.npy`. BRICS graphs are built from SMILES during data loading; ProtT5 residue features are generated from protein sequences during data loading.

Set `PROTT5_PATH` to a local ProtT5 model directory. Feature extraction uses MolT5 and ESM-2 weights; set `MOLT5_PATH` and `ESM2_PATH` if their local directories differ from the defaults in `paths.py`. Pretrained weights, cached data, and trained checkpoints are not included in this source directory.

## Training

Run commands from this source directory. To train the current architecture from a new model initialization:

```bash
export PROTT5_PATH=/path/to/prot_t5_xl_local
python main.py --cfg configs/RPCG-EDL.yaml --data kcatkm
```

The saved final checkpoint was obtained by fine-tuning from a fragment-graph checkpoint at epoch 62. To follow that initialization path, supply the earlier checkpoint explicitly:

```bash
python main.py \
  --cfg configs/RPCG-EDL_finetune20.yaml \
  --data kcatkm \
  --pretrained-path /path/to/fragment_graph_epoch62.pth
```

## Independent evaluation

```bash
python independent_test_attention.py \
  --cfg configs/independent_test.yaml \
  --data independent_test \
  --model-path /path/to/rpcg_edl_epoch13.pth \
  --output-dir result/independent_test_806
```

`requirements.txt` lists the Python packages imported by the source. The local validation environment used Python 3.9, PyTorch 2.8, DGL 2.4, and Transformers 4.57. GPU builds of PyTorch and DGL must match the target CUDA environment. The license is in `LICENSE`.
