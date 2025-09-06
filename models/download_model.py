from sentence_transformers import SentenceTransformer

def download_and_save(model_name: str, save_path: str):
    """
    Download a SentenceTransformer model and save it locally.
    """
    print(f" Downloading model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f" Saving model to: {save_path}")
    model.save(save_path)
    print(" Done.")

if __name__ == "__main__":
    # Example: all-MiniLM-L6-v2
    download_and_save("sentence-transformers/all-MiniLM-L6-v2", "./all-MiniLM-L6-v2")
