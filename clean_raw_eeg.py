import os
import glob
import mne
from mne_icalabel import label_components
import pandas as pd
import numpy as np

# ==========================================
# 1. SETUP PATHS
# ==========================================
root_dir = '/home/stud1/Desktop/Swathi/COGBCI_data_codes/Neuroflow/raw_data'

# 🚨 THE UPGRADE: Automatically find EVERY subject folder in raw_data
# This ignores hidden files/folders and sorts them neatly
subjects = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')]
subjects.sort()

print(f"🔍 Scanning {root_dir}...")
print(f"✅ Found {len(subjects)} subjects to process.\n")

# ==========================================
# 2. PROCESSING LOOP
# ==========================================
for subj in subjects:
    subject_path = os.path.join(root_dir, subj)
    
    # Find the EDF recursively
    edf_files = glob.glob(os.path.join(subject_path, '**', '*.edf'), recursive=True)
    if not edf_files:
        print(f"⚠️ No EDF found for Subject: {subj}")
        continue
        
    # Pick the largest file to avoid tiny calibration/aborted files
    current_edf = max(edf_files, key=os.path.getsize)
    print(f"\n=========================================================")
    print(f"⚙️ Processing Subject {subj}: {os.path.basename(current_edf)}")
    
    try:
        # Load Data
        print("   -> Loading EDF...")
        raw = mne.io.read_raw_edf(current_edf, preload=True, verbose=False)
        
        # SMART CHANNEL SELECTION (Handles MATLAB/Python naming differences)
        available_chans = raw.ch_names
        target_channels = {
            'Fz': ['FZ', 'Fz', 'fz'],
            'Cz': ['CZ', 'Cz', 'cz'],
            'C3': ['C3', 'c3'],
            'C4': ['C4', 'c4'],
            'O1': ['O1', 'o1'],
            'O2': ['O2', 'o2'],
            'F3': ['F3(BLUE)', 'F3_BLUE_', 'F3-BLUE', 'F3_BLUE', 'F3', 'f3'],
            'F4': ['F4(RED)', 'F4_RED_', 'F4-RED', 'F4_RED', 'F4', 'f4']
        }
        
        channel_mapping = {}
        missing_targets = []
        
        for std_name, variations in target_channels.items():
            match = next((var for var in variations if var in available_chans), None)
            if match:
                channel_mapping[match] = std_name
            else:
                missing_targets.append(std_name)
                
        if missing_targets:
            raise ValueError(f"Missing {missing_targets}. Available: {available_chans}")
            
        # Drop unwanted channels and rename the 8 we need
        raw.pick(list(channel_mapping.keys()))
        raw.rename_channels(channel_mapping)
        
        # Set standard 10-20 montage coordinates
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)
        
        # Filter & Resample
        print("   -> Filtering (0.5 - 35 Hz) and Resampling to 500Hz...")
        raw.filter(l_freq=0.5, h_freq=35.0, verbose=False)
        raw.resample(500.0, verbose=False)
        
        # Run ICA and Auto-Reject Artifacts
        print("   -> Running ICA...")
        ica = mne.preprocessing.ICA(n_components=len(raw.ch_names), random_state=42, max_iter='auto')
        ica.fit(raw, verbose=False)
        
        ic_labels = label_components(raw, ica, method='iclabel')
        
        bad_comps = [idx for idx, label in enumerate(ic_labels['labels']) if label != 'brain']
        ica.exclude = bad_comps
        
        print(f"   -> Removed {len(bad_comps)} non-brain components.")
        
        clean_raw = ica.apply(raw.copy(), verbose=False)
        
        # Average Reference
        clean_raw.set_eeg_reference('average', projection=False, verbose=False)
        
        # Save to CSV [Time x 8 Channels]
        save_path = current_edf.replace('.edf', '_cleaned_EEG_500Hz.csv')
        data_array = clean_raw.get_data()
        
        pd.DataFrame(data_array.T).to_csv(save_path, index=False, header=False)
        print(f"   ✅ Successfully Saved: {save_path}")

    except Exception as e:
        print(f"   ❌ FAILED on Subject {subj}: {e}")

print("\n🎉 ALL SUBJECTS PROCESSED!")
