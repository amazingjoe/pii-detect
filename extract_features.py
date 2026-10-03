import sys
import argparse
from typing import List, Optional, Union
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from settings import add_selection_args, load_settings, preparse_selection


class FeatureExtractor:
    def __init__(self, model_path: str, device: str = None, max_layer: Optional[int] = None):
        """Loads the base model (no LM head: only hidden states are used, so vocabulary logits are never computed).

        Args:
            model_path: Local path to the model.
            device: Torch device; defaults to MPS when available, otherwise CPU.
            max_layer: If set, transformer layers above this hidden_states index are dropped, so a
                forward pass stops at the layer a probe reads. Hidden states up to max_layer are
                identical to the full model's. Leave unset when several layers are needed (training,
                the layer sweep).
        """
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using compute device: {self.device}", file=sys.stderr)
        print(f"Loading tokenizer and model from '{model_path}'...", file=sys.stderr)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModel.from_pretrained(model_path, dtype=torch.float16)
        self.full_num_layers = self.model.config.num_hidden_layers
        self.max_layer = None
        if max_layer is not None:
            self._truncate(self.resolve_layer(max_layer))
        self.model.to(self.device)
        self.model.eval()

    def resolve_layer(self, layer_index: int) -> int:
        """Converts a hidden_states index (negative counts from the top of the full model) to a positive one."""
        resolved = layer_index + self.full_num_layers + 1 if layer_index < 0 else layer_index
        if not 0 <= resolved <= self.full_num_layers:
            raise ValueError(
                f"Layer {layer_index} is out of range for a model with {self.full_num_layers} layers."
            )
        return resolved

    def _truncate(self, layer: int):
        if layer < self.full_num_layers:
            self.model.layers = self.model.layers[:layer]
            self.model.config.num_hidden_layers = layer
            self.model.config.layer_types = self.model.config.layer_types[:layer]
            # transformers replaces the top hidden state with the final-normed output. The full model's
            # hidden_states[layer] is un-normed for any layer below the top, so drop the norm to match it.
            self.model.norm = torch.nn.Identity()
            print(f"Stopping forward passes at layer {layer} of {self.full_num_layers}.", file=sys.stderr)
        self.max_layer = layer

    def extract(self, text: Union[str, List[str]], layer_index: int) -> np.ndarray:
        """Extract mean-pooled hidden state representation(s) for text or a batch of texts.
        
        Args:
            text: A single string or list of strings.
            layer_index: The transformer layer index to extract from (e.g. -2, second-to-last layer).
            
        Returns:
            np.ndarray: Vector of shape (hidden_dim,) for a single string, 
                        or (batch_size, hidden_dim) for a list of strings.
        """
        if self.max_layer is not None and not 0 <= layer_index <= self.max_layer:
            # A negative index would count from the top of the truncated model, not the full one
            raise ValueError(
                f"Layer {layer_index} is not available: this extractor stops at layer {self.max_layer} "
                f"and needs a positive index (use resolve_layer)."
            )
        is_single = isinstance(text, str)
        if is_single:
            text = [text]

        hidden_states, attention_mask = self._forward(text)
        pooled = self._mean_pool(hidden_states[layer_index], attention_mask)

        return pooled[0] if is_single else pooled

    def extract_all_layers(self, texts: List[str]) -> np.ndarray:
        """Extract mean-pooled representations from every hidden state in one forward pass.

        Index 0 is the embedding output and index i is the output of transformer layer i,
        matching the indexing of `extract(..., layer_index=i)`.

        Returns:
            np.ndarray: Array of shape (num_hidden_states, batch_size, hidden_dim).
        """
        hidden_states, attention_mask = self._forward(texts)
        return np.stack([self._mean_pool(layer, attention_mask) for layer in hidden_states])

    def _forward(self, texts: List[str]):
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        return outputs.hidden_states, inputs["attention_mask"]

    @staticmethod
    def _mean_pool(layer: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
        # Mask-aware mean pooling (ignores padding tokens when averaging)
        mask = attention_mask.unsqueeze(-1).expand(layer.size()).float()
        sum_embeddings = torch.sum(layer * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        return (sum_embeddings / sum_mask).cpu().float().numpy()


if __name__ == "__main__":
    settings = load_settings()
    sel = preparse_selection(settings)

    parser = argparse.ArgumentParser(
        description="Extract feature representations from text using an LLM hidden state."
    )
    add_selection_args(parser, settings, head=False)
    parser.add_argument(
        "text",
        type=str,
        help="Text to extract features from (required, no fallback)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=sel["model_path"],
        help="Path to the model directory, overriding --model (default: the selected model's path)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=sel["layer"],
        help="Hidden state layer index to extract (default: the selected model's layer)",
    )
    args = parser.parse_args()

    extractor = FeatureExtractor(model_path=args.model_path)
    vector = extractor.extract(args.text, layer_index=args.layer)

    print("\n--- Extracted Representation ---")
    print(f"Final vector dimensions: {vector.shape}")
    print(
        f"Vector summary: min={vector.min():.4f}, max={vector.max():.4f}, mean={vector.mean():.4f}"
    )
    print(f"First 10 values:\n{vector[:10]}")