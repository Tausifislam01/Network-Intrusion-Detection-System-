from pathlib import Path
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

REPO_ID = "c01dsnap/CIC-IDS2017"
REPO_TYPE = "dataset"

# Corrected exact filenames from the Hugging Face repository
FILES_TO_DOWNLOAD = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]

def ensure_raw_dir():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_file(filename):
    destination = RAW_DATA_DIR / filename
    if destination.exists():
        print(f"Already exists: {destination}")
        return destination

    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type=REPO_TYPE,
    )
    destination.write_bytes(Path(downloaded_path).read_bytes())
    print(f"Downloaded: {destination}")
    return destination

def download_dataset():
    ensure_raw_dir()
    downloaded_files = []
    for filename in FILES_TO_DOWNLOAD:
        try:
            path = download_file(filename)
            downloaded_files.append(path)
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
    return downloaded_files

if __name__ == "__main__":
    download_dataset()