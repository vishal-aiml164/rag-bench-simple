import os
import logging
import pandas as pd
from datasets import load_dataset

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_dataset(dataset_name="wikitext", subset="wikitext-2-raw-v1", split_ratios=None, data_dir="data/input"):
    """
    Download dataset from HuggingFace on first run, save as CSV, and reuse from CSV after that.
    Returns a dictionary of pandas DataFrames (train/validation/test).
    """

    os.makedirs(data_dir, exist_ok=True)
    dataset_files = {
        "train": os.path.join(data_dir, f"{dataset_name}_train.csv"),
        "validation": os.path.join(data_dir, f"{dataset_name}_validation.csv"),
        "test": os.path.join(data_dir, f"{dataset_name}_test.csv"),
    }

    dataframes = {}

    # Check if CSVs already exist
    if all(os.path.exists(f) for f in dataset_files.values()):
        logger.info("Loading dataset from local CSV files...")
        for split, file in dataset_files.items():
            dataframes[split] = pd.read_csv(file)
    else:
        logger.info(f"Downloading dataset {dataset_name}/{subset} from HuggingFace...")
        dataset = load_dataset(dataset_name, subset)

        # Convert splits to pandas and save to CSV
        for split in ["train", "validation", "test"]:
            if split in dataset:
                df = dataset[split].to_pandas()
                df.to_csv(dataset_files[split], index=False)
                logger.info(f"Saved {split} split to {dataset_files[split]}")
                dataframes[split] = df

    return dataframes
