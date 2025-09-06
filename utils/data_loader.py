import os
import logging
import pandas as pd
from datasets import load_dataset, DatasetDict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_dataset(dataset_name="msmarco", subset=None, data_dir="data/input"):
    """
    Download dataset from HuggingFace on first run, save as CSV, and reuse from CSV after that.
    Returns a dictionary of pandas DataFrames (train/validation/test).
    
    Supported dataset_name values:
      - "squad"                     → Stanford Question Answering Dataset
      - "msmarco"                   → MS MARCO passage ranking
      - "sentence-transformers/msmarco-msmarco-MiniLM-L6-v3" → A pre-processed MS MARCO
      - "natural_questions"         → Google Natural Questions
      - "beir"                      → Benchmark of 18 IR datasets (subset required, e.g. "scifact")
      - "wikitext" (legacy, default for testing)
    """
    os.makedirs(data_dir, exist_ok=True)

    dataset_key = f"{dataset_name.replace('/', '_')}_{subset or 'default'}"
    dataset_files = {
        "train": os.path.join(data_dir, f"{dataset_key}_train.csv"),
        "validation": os.path.join(data_dir, f"{dataset_key}_validation.csv"),
        "test": os.path.join(data_dir, f"{dataset_key}_test.csv"),
    }

    dataframes = {}

    # Reuse if CSVs already exist
    if all(os.path.exists(f) for f in dataset_files.values()):
        logger.info(f"Loading {dataset_key} from local CSV files...")
        for split, file in dataset_files.items():
            if os.path.exists(file):
                dataframes[split] = pd.read_csv(file)
            else:
                dataframes[split] = None
        return dataframes

    # Otherwise download from HuggingFace
    logger.info(f"Downloading dataset {dataset_name}/{subset} from HuggingFace...")

    dataset = None
    if dataset_name == "squad":
        # SQuAD's common subset is "plain_text"
        dataset = load_dataset(dataset_name, "plain_text")
    elif dataset_name == "msmarco":
        dataset = load_dataset("ms_marco", subset)
    elif dataset_name == "sentence-transformers/msmarco-msmarco-MiniLM-L6-v3":
        # Pass the subset to the load_dataset function
        dataset = load_dataset(dataset_name, subset)
    elif dataset_name == "natural_questions":
        dataset = load_dataset("natural_questions", "default")
    elif dataset_name == "beir":
        if not subset:
            raise ValueError("BEIR requires a subset (e.g., scifact, nfcorpus, fiqa).")
        dataset = load_dataset("beir", subset)
    elif dataset_name == "wikitext":
        dataset = load_dataset("wikitext", subset or "wikitext-2-raw-v1")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if not isinstance(dataset, DatasetDict):
        dataset_dict = DatasetDict({
            split: ds for split, ds in dataset.items()
        })
        dataset = dataset_dict

    # Convert splits to pandas and save to CSV
    for split in ["train", "validation", "test"]:
        if split in dataset:
            df = dataset[split].to_pandas()
            df.to_csv(dataset_files[split], index=False)
            logger.info(f"Saved {split} split to {dataset_files[split]}")
            dataframes[split] = df
        else:
            logger.warning(f"{split} split not found for {dataset_name}")
            dataframes[split] = None

    return dataframes
