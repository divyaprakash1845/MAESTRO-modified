Before running the Python scripts, install the required dependencies:

```bash
pip install torch pandas numpy scipy mne

```

## 🚀 Execution Steps

### 1. EEG Artifact Removal (MATLAB)

Cleans raw `.edf` files using ICA and selects the largest recording automatically.

* **Command:** Open and Run `clean_raw_eeg.m` in MATLAB.
* **Note:** Update the `eegpath` (EEGLAB folder) and `rootDir` (your `raw_data` folder) at the top of the script before running.
* Clone the repository
```bash
git clone https://github.com/divyaprakash1845/MAESTRO-modified
%cd MAESTRO-modified

```

### 2. Multi-Modal Fusion (Python)

```bash
python preprocess.py

```

### 3. Model Training (Python)

```bash
python train.py

```

---

## 📁 Required Folder Structure

```text
Workspace/
├── raw_data/                 <-- Raw lab folders (7873, etc.)
└── MAESTRO-modified/          <-- (This Repository)
    ├── clean_raw_eeg.m
    ├── preprocess.py
    ├── dataset.py
    ├── model.py
    └── train.py

```
