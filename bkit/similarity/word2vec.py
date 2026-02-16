import os
import glob
import gensim
import huggingface_hub


class BanglaWord2Vec(gensim.models.KeyedVectors):
    """
    Automatically loads Bangla Word2Vec from Hugging Face when instantiated.
    """

    def __init__(self, repo_id: str = "sayedshaungt/bangla-word2vec-300d"):
        # Download repository snapshot
        local_dir = huggingface_hub.snapshot_download(
            repo_id=repo_id, repo_type="model"
        )

        # Locate .embeddings file
        emb_files = glob.glob(os.path.join(local_dir, "*.embeddings"))
        if not emb_files:
            raise FileNotFoundError(f"No .embeddings file found in {local_dir}")

        # Load KeyedVectors normally
        kv = gensim.models.KeyedVectors.load(emb_files[0])

        # Copy internal state into `self`
        self.__dict__.update(kv.__dict__)
