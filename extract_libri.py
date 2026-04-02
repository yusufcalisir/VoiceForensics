import tarfile
import os
import shutil

os.makedirs("sample_dataset", exist_ok=True)
count_speakers = 0

with tarfile.open("dev-clean.tar.gz", "r:gz") as tar:
    members = tar.getmembers()
    for m in members:
        # e.g. LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac
        if m.name.endswith(".flac"):
            parts = m.name.split('/')
            speaker_id = parts[2]
            
            # create speaker group
            spk_dir = os.path.join("sample_dataset", f"Speaker_{speaker_id}")
            if not os.path.exists(spk_dir):
                if count_speakers >= 5:
                    continue
                os.makedirs(spk_dir, exist_ok=True)
                count_speakers += 1
                
            # copy only 2 files per speaker
            if len(os.listdir(spk_dir)) < 2:
                # extract
                tar.extract(m, path="temp_extract")
                # move to sample_dataset
                shutil.move(os.path.join("temp_extract", m.name), os.path.join(spk_dir, parts[-1]))
                
print("Extracted subsets successfully to sample_dataset!")
