"""Intent Taxonomy Clustering Script for AppleSupport.

Performs offline unsupervised clustering on initial customer queries:
1. Samples 2,000 historical customer tweets.
2. Generates local embeddings via LocalEmbeddingModel.
3. Evaluates silhouette scores across k in [6, 12].
4. Extracts representative samples and frequent keywords for each cluster.
"""

from __future__ import annotations

import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import re
import random
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from src.utils.embeddings import LocalEmbeddingModel

random.seed(42)
np.random.seed(42)

HISTORICAL_PATH = "data/processed/historical_kb.jsonl"
TAXONOMY_DIR = "src/taxonomy"

def clean_tweet_for_clustering(text: str) -> str:
    # Remove @mentions and URLs for pure semantic clustering
    t = re.sub(r'@\w+', '', text)
    t = re.sub(r'https?://\S+', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def run_clustering():
    print("Loading historical customer queries for clustering...")
    samples = []
    with open(HISTORICAL_PATH) as f:
        for line in f:
            data = json.loads(line)
            clean_txt = clean_tweet_for_clustering(data["customer_text"])
            if len(clean_txt) >= 20: # ignore ultra-short chatter
                samples.append({
                    "tweet_id": data["customer_tweet_id"],
                    "raw_text": data["customer_text"],
                    "clean_text": clean_txt
                })
                
    print(f"Loaded {len(samples):,} valid candidate queries.")
    
    # Sample 2,000 queries
    sample_size = min(2000, len(samples))
    sampled_queries = random.sample(samples, sample_size)
    print(f"Sampled {sample_size} queries for embedding & clustering.")
    
    # Generate local embeddings
    print("Generating dense embeddings using local BGE model...")
    embedder = LocalEmbeddingModel()
    texts_to_embed = [q["clean_text"] for q in sampled_queries]
    embeddings = embedder.embed_texts(texts_to_embed, batch_size=64)
    print(f"Embeddings generated: shape {embeddings.shape}")
    
    # Sweep k = 6 to 12
    print("\nEvaluating silhouette scores over k in [6..12]...")
    best_k = 8
    best_score = -1.0
    results = {}
    
    for k in range(6, 13):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = kmeans.fit_predict(embeddings)
        score = float(silhouette_score(embeddings, labels))
        results[k] = score
        print(f"  k={k:2d}: silhouette_score = {score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
            
    print(f"\nBest cluster count: k={best_k} (score={best_score:.4f})")
    
    # Fit final clustering with best_k (or k=8 for optimal balance of specificity)
    final_k = 8
    kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    clusters_info = []
    for c_id in range(final_k):
        c_mask = (labels == c_id)
        c_indices = np.where(c_mask)[0]
        c_size = len(c_indices)
        
        # Find nearest examples to centroid
        centroid = kmeans.cluster_centers_[c_id]
        c_embeds = embeddings[c_indices]
        dists = np.linalg.norm(c_embeds - centroid, axis=1)
        top_idx_in_cluster = np.argsort(dists)[:8]
        representative_examples = [sampled_queries[c_indices[i]]["raw_text"] for i in top_idx_in_cluster]
        
        # Word frequency in cluster
        all_words = " ".join([sampled_queries[i]["clean_text"].lower() for i in c_indices])
        tokens = re.findall(r'\b[a-z]{3,}\b', all_words)
        stopwords = {'the', 'and', 'for', 'that', 'this', 'with', 'you', 'have', 'not', 'are', 'can', 'was', 'but', 'out', 'all', 'any'}
        keywords = [w for w, count in Counter(tokens).most_common(20) if w not in stopwords][:7]
        
        clusters_info.append({
            "cluster_id": c_id,
            "size": c_size,
            "pct": round(c_size / sample_size * 100, 1),
            "keywords": keywords,
            "representative_examples": representative_examples
        })
        
    print("\n--- CLUSTER PROFILES ---")
    for c in clusters_info:
        print(f"\nCluster #{c['cluster_id']} (size: {c['size']}, {c['pct']}%):")
        print(f"  Keywords: {', '.join(c['keywords'])}")
        print("  Sample queries:")
        for ex in c['representative_examples'][:2]:
            print(f"    - {ex}")
            
    # Save clustering diagnostics
    diag_path = "data/clustering_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump({
            "sweep_results": results,
            "best_k": best_k,
            "final_k": final_k,
            "clusters": clusters_info
        }, f, indent=2)
    print(f"\nDiagnostics saved to {diag_path}")

if __name__ == "__main__":
    run_clustering()
