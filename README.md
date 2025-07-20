
# HPDAF-DTA

## 🌟 Introduction

**HPDAF-DTA (Hierarchically Progressive Dual-Attention Fusion for Drug–Target Affinity Prediction)** is a novel multimodal deep learning model designed to accurately predict drug–target binding affinity (DTA). This method comprehensively integrates three types of biochemical data: protein sequences, drug molecular graphs, and protein–drug structural interaction graphs (Pocket–Drug Graph, PD graph).

Compared with existing approaches, HPDAF-DTA offers the following key advantages:

- 🧠 **Dual-Attention Fusion Mechanism**: Captures both intra-modality dependencies and cross-modality complementarities to achieve fine-grained feature fusion.
- ⚛️ **Structure-Aware Graph Modeling**: Introduces the Pocket–Drug graph to encode spatial interactions between drugs and targets, addressing limitations of sequence-only or flat graph-based methods.
- 📈 **End-to-End Regression Framework**: Jointly learns multimodal representations of drugs and targets and directly predicts continuous affinity values (e.g., pKd, pKi).

The model achieves state-of-the-art performance on multiple benchmark datasets such as CASF-2016 and CASF-2013, demonstrating its practical utility in virtual screening and drug discovery tasks.

---

## 🔧 Installation & Usage

```bash
# Clone the repository
git clone https://github.com/BioinfoYB/HPDAF-DTA.git
cd HPDAF-DTA

# Create environment from YAML
conda env create -f environment.yaml
conda activate hpdaf
```

---

## 🚀 Training & Testing

To train the model and evaluate it on the test set (e.g., CASF-2016), run:

```bash
python main.py
```

After training, you will obtain:

- Model weights in `./result/`
- Evaluation results on the test set in `./result/`

---

## 📦 Preprocessing Raw Data

To convert raw PDBbind-format data into the required input format, use the script `convert_data.py`:

```bash
python convert_data.py \
  --input_pdb ./raw_data \
  --output ./processed_data
```

This will generate `processed_data.pt` in PyTorch format.

Expected input folder structure:

```
raw_data/
├── 1abc/
│   ├── protein.pdb
│   ├── ligand.smi
│   └── affinity.txt
...
```

---

## 📚 Dataset Source

The structural data used in this project comes from the official PDBbind database:

🔗 http://pdbbind.org.cn/

We use:
- The general and refined sets from PDBbind v2020 for training and validation.
- CASF-2016 and CASF-2013 benchmark sets for testing.

For detailed data processing steps, refer to the supplementary documentation or the `convert_data.py` script.

---
