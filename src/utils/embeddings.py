"""Local ONNX-based dense text embedding generator using bge-small-en-v1.5.

Runs locally with zero external API calls, optimized for Apple Silicon / CPU.
Embeddings are normalized 384-dimensional dense vectors.
"""

import os
import json
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

class LocalEmbeddingModel:
    def __init__(self, model_dir: str = "models/bge-small"):
        self.model_dir = model_dir
        onnx_path = os.path.join(model_dir, "model_optimized.onnx")
        tokenizer_path = os.path.join(model_dir, "tokenizer.json")
        
        if not os.path.exists(onnx_path) or not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Missing ONNX model or tokenizer in {model_dir}")
        
        # Fast dynamic tokenizer (tweets are <128 tokens)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(max_length=128)
        self.tokenizer.enable_padding(pad_to_multiple_of=8)
        
        # Load ONNX runtime session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])
        
    def embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Generate normalized 384-d embeddings for a list of text strings."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
            
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self.tokenizer.encode_batch(batch)
            
            input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
            token_type_ids = np.zeros_like(input_ids)
            
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            }
            
            # Run inference
            outputs = self.session.run(None, inputs)
            # outputs[0] is typically last_hidden_state [batch, seq_len, 384]
            last_hidden = outputs[0]
            
            # Mean pooling with attention mask
            mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
            sum_embeddings = np.sum(last_hidden * mask_expanded, axis=1)
            sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
            mean_pooled = sum_embeddings / sum_mask
            
            # L2 normalization for cosine similarity
            norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
            norms = np.clip(norms, a_min=1e-9, a_max=None)
            normalized = mean_pooled / norms
            all_embeddings.append(normalized)
            
        return np.vstack(all_embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query text."""
        return self.embed_texts([query])[0]


if __name__ == "__main__":
    embedder = LocalEmbeddingModel()
    test_queries = [
        "My iPhone battery drains within 2 hours",
        "How do I reset my Apple ID password?",
        "My MacBook battery is dying very quickly"
    ]
    vectors = embedder.embed_texts(test_queries)
    print(f"Generated embeddings shape: {vectors.shape}")
    
    # Cosine similarity check
    sim_0_1 = np.dot(vectors[0], vectors[1])
    sim_0_2 = np.dot(vectors[0], vectors[2])
    print(f"Similarity (Battery iPhone vs Apple ID): {sim_0_1:.4f}")
    print(f"Similarity (Battery iPhone vs Battery Mac): {sim_0_2:.4f}")
    assert sim_0_2 > sim_0_1, "Semantic similarity failed"
    print("Local embedding validation passed successfully!")
