import os
from pathlib import Path
import bkit
from bkit.utils import MODEL_URL_MAP, load_cached_file
from ._helpers import Infer_Pos_Noisy
import numpy as np
from spacy import displacy
import spacy
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
import logging


def visualize(pos_data):
    """
    Visualize POS tags using spaCy's displacy with custom colors
    
    Args:
        pos_data: List of tuples containing (word, pos_tag, confidence)
    """
    colors = {
        "PRO": "blueviolet",
        "VF": "lightpink",
        "NNC": "turquoise", 
        "ADJ": "lime",
        "NNP": "khaki",
        "ADV": "orange",
        "CONJ": "cornflowerblue",
        "DET": "forestgreen",
        "QF": "salmon",
        "VNF": "yellow",
        "PUNCT": "gray",
        "PP": "lightblue",
        "PART": "pink",
        "INTJ": "lightgreen"
    }

    text = " ".join([word for word, _, _ in pos_data])
    
    spans = []
    current_pos = 0
    for word, tag, _ in pos_data:
        start = current_pos
        end = start + len(word)
        spans.append((start, end, tag))
        current_pos = end + 1  # +1 for space

    # Create displacy document
    doc = {
        "text": text,
        "ents": [{"start": start, "end": end, "label": tag} 
                 for start, end, tag in spans]
    }

    # Visualization options
    options = {
        "ents": list(colors.keys()),
        "colors": colors
    }

    # Render
    displacy.render(doc, style="ent", options=options, manual=True)



class HuggingFaceModel:
    logging.getLogger("transformers").setLevel(logging.ERROR)

    """"
    A wrapper class for HuggingFace token classification models, designed for part-of-speech (POS) tagging
    or similar sequence labeling tasks.
    This class provides methods to load a pretrained HuggingFace model and tokenizer, perform predictions
    on input text, and align subword tokens to reconstruct original words with their predicted labels.

    Attributes:
        model: The HuggingFace model instance for token classification.
        tokenizer: The HuggingFace tokenizer instance.
        labels: The list of label names used for classification.
    Methods:
        from_pretrained(model_name, **kwargs):
            Class method to load a pretrained model and tokenizer by name, with optional custom labels.
        predict(text: str) -> list:
            Predicts token labels for the given input text and returns a list of aligned word-label-score tuples.
    """
 
    def __init__(self, **kwargs) -> None:
        self.model = kwargs.get("model")
        self.tokenizer = kwargs.get("tokenizer")
        self.labels = kwargs.get("labels")

    @classmethod
    def from_pretrained(cls, model_name, **kwargs) -> 'HuggingFaceModel':
        DEFAULT_LABELS = [
            'PP','DET','INTJ','VNF','OTH','NNP','QF','CONJ','ADV','ADJ','NNC',
            'PRO','PART','PUNCT','VF'
            ]
        if kwargs.get("labels") is None:
            labels = DEFAULT_LABELS
        else:
            assert isinstance(kwargs.get("labels"), list), "labels must be a list"
            labels = kwargs.get("labels")
        
        label_to_id = {l: i for i, l in enumerate(labels)}
        id_to_label = {i: l for l, i in label_to_id.items()}

        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(labels),
            id2label=id_to_label,
            label2id=label_to_id,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return cls(model=model, tokenizer=tokenizer, labels=labels)


    def __call__(self, text: str) -> list:
        nlp = pipeline("token-classification", model=self.model, tokenizer=self.tokenizer)
        return self._align_tokens(nlp(text))

    def _align_tokens(self, tokens: list) -> list[tuple]:
        words = []
        curr_word, curr_label, curr_score = None, None, None

        for t in tokens:
            token, label, score = t["word"], t["entity"], np.array(t["score"]).item()

            if token.startswith("##"):
                curr_word += token[2:]
            else:
                if curr_word is not None:
                    words.append((curr_word, curr_label, curr_score))
                curr_word, curr_label, curr_score = token, label, score

        if curr_word is not None:
            words.append((curr_word, curr_label, curr_score))
        return words


class Infer:
    """
    Args:
        model_name (str): Name of the pre-trained POS model.
        pretrained_model_path (str): Path to a custom pre-trained model directory.
        force_redownload (bool): Flag to force redownload the cached model.

    Attributes:
        model_name (str): Name of the pre-trained POS model.
        infer_class (Infer_Noisy): Inference class for noisy label model or other POS models.
        model_cache_dir (Path): Path to the cached model files.

    Methods:
        __init__(self, model_name: str = None, pretrained_model_path: str = None, force_redownload: bool = False) -> None:
            Initializes the Infer class instance.

        infer(self, text: str = '...') -> dict:
            Perform POS inference on the input text.

        @staticmethod
        from_huggingface(model_name, **kwargs) -> HuggingFaceModel:
            Loads a HuggingFace model by its name.

    Example:
    ```python
    # Load noisy label architecture
    from bkit.pos import Infer
    model = Infer("pos-noisy-label")
    model("কাতার বিশ্বকাপে আর্জেন্টিনার বিশ্বকাপ জয়ে মার্তিনেজের অবদান অনেক।")

    # Load huggingface architecture
    from bkit.pos import Infer
    model = Infer.from_huggingface("local_path_or_huggingface_model_name")
    model("কাতার বিশ্বকাপে আর্জেন্টিনার বিশ্বকাপ জয়ে মার্তিনেজের অবদান অনেক।")
    ```
    """

    def __init__(
        self,
        model_name: str = None,
        pretrained_model_path: str = None,
        force_redownload: bool = False,
    ) -> None:
        """
        Initializes the Infer class instance.

        Args:
            model_name (str): Name of the pre-trained POS model.
            pretrained_model_path (str): Path to a custom pre-trained model directory.
            force_redownload (bool): Flag to force redownload if the model is not cached.
        """
        self.model_name = model_name
        cache_dir = Path(bkit.ML_MODELS_CACHE_DIR)
        model_cache_dir = cache_dir / f"{model_name}"

        if self.model_name == "pos-noisy-label":
            self.infer_class = Infer_Pos_Noisy

        if pretrained_model_path and os.path.exists(pretrained_model_path):
            self.infer_class = self.infer_class(pretrained_model_path)
        else:
            if model_name not in MODEL_URL_MAP and not os.path.exists(model_cache_dir):
                raise Exception(
                    "model name not found in download map and no model cache directory found"
                )

            # load the model files from cache directory or force download and save in cache
            self.model_cache_dir = load_cached_file(
                model_name, force_redownload=force_redownload
            )
            self.infer_class = self.infer_class(self.model_cache_dir)

    def __call__(self, text: str) -> dict:
        """
        Perform POS inference on the input text.

        Args:
            text (str): Input text for POS inference.

        Returns:
            dict: Dictionary containing POS inference results.
        """
        result = self.infer_class.infer(text)
        return result

    @staticmethod
    def from_huggingface(model_name, **kwargs) -> 'HuggingFaceModel':
        """
        Loads a HuggingFace model by its name.
        Parameters:
            model_name (str): The name or path of the pretrained model to load from HuggingFace.
            **kwargs: Additional keyword e.g. labels.

        Returns:
            HuggingFaceModel: An instance of HuggingFaceModel loaded with the specified pretrained weights.

        Example:
        ```python
        from bkit.pos import Infer
        model = Infer.from_huggingface("local_path_or_huggingface_model_name")
        model("কাতার বিশ্বকাপে আর্জেন্টিনার বিশ্বকাপ জয়ে মার্তিনেজের অবদান অনেক।")
        """
        return HuggingFaceModel.from_pretrained(model_name, **kwargs)