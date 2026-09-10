"""Thread Reconstruction Pipeline for AppleSupport.

Extracts complete conversation chains:
  customer_query -> brand_reply -> customer_followup (if any)

Applies strict temporal partitioning:
  - 80% Historical KB & Training Pool (older time-slice)
  - 20% Held-out Evaluation Pool (recent time-slice for Golden Set)
"""

from __future__ import annotations

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

DATA_PATH = "data/kaggle-dataset/twcs/twcs.csv"
OUTPUT_DIR = "data/processed"
BRAND = "AppleSupport"

def run_thread_reconstruction():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Starting Thread Reconstruction for brand: {BRAND}...")
    
    # 1. Collect all outbound tweets from AppleSupport
    apple_outbound_list = []
    inbound_tweet_ids = set()
    
    chunk_size = 500_000
    print("Pass 1: Extracting AppleSupport outbound tweets and parent IDs...")
    for chunk in pd.read_csv(DATA_PATH, chunksize=chunk_size):
        mask = (chunk['author_id'] == BRAND) & (chunk['inbound'] == False)
        sub = chunk[mask]
        if len(sub) > 0:
            apple_outbound_list.append(sub)
            valid_in = sub['in_response_to_tweet_id'].dropna().astype(int)
            inbound_tweet_ids.update(valid_in.tolist())
            
    outbound_df = pd.concat(apple_outbound_list, ignore_index=True)
    print(f"  Found {len(outbound_df):,} outbound tweets from {BRAND}.")
    print(f"  Target customer root tweets to find: {len(inbound_tweet_ids):,}")
    
    # Also collect outbound tweet IDs to look for customer followups
    brand_reply_ids = set(outbound_df['tweet_id'].tolist())
    
    # 2. Extract customer initial tweets and followups in Pass 2
    print("Pass 2: Extracting matching customer tweets & followups...")
    matching_inbound = []
    matching_followups = []
    
    for chunk in pd.read_csv(DATA_PATH, chunksize=chunk_size):
        # Root customer tweets
        in_mask = chunk['tweet_id'].isin(inbound_tweet_ids)
        if in_mask.any():
            matching_inbound.append(chunk[in_mask])
            
        # Followups to brand replies (where in_response_to_tweet_id is a brand reply)
        valid_resp = chunk['in_response_to_tweet_id'].dropna().astype(int)
        followup_mask = (chunk['inbound'] == True) & (valid_resp.isin(brand_reply_ids))
        if followup_mask.any():
            matching_followups.append(chunk[followup_mask])
            
    inbound_df = pd.concat(matching_inbound, ignore_index=True)
    followup_df = pd.concat(matching_followups, ignore_index=True) if matching_followups else pd.DataFrame()
    print(f"  Found {len(inbound_df):,} parent customer tweets.")
    print(f"  Found {len(followup_df):,} customer follow-up tweets.")
    
    # 3. Clean and merge
    # Sort outbounds by created_at and drop duplicate replies to same tweet, keeping first reply
    outbound_df['dt_brand'] = pd.to_datetime(outbound_df['created_at'], format="%a %b %d %H:%M:%S %z %Y")
    outbound_df = outbound_df.sort_values('dt_brand').drop_duplicates(subset=['in_response_to_tweet_id'])
    
    inbound_df['dt_cust'] = pd.to_datetime(inbound_df['created_at'], format="%a %b %d %H:%M:%S %z %Y")
    
    # Merge customer root + brand reply
    merged = outbound_df.merge(
        inbound_df,
        left_on='in_response_to_tweet_id',
        right_on='tweet_id',
        suffixes=('_brand', '_cust')
    )
    
    # Map followups (if any) to brand_reply tweet_id
    followup_map = {}
    if not followup_df.empty:
        followup_df['dt_follow'] = pd.to_datetime(followup_df['created_at'], format="%a %b %d %H:%M:%S %z %Y")
        # Keep earliest followup per brand reply
        followup_sorted = followup_df.sort_values('dt_follow').drop_duplicates(subset=['in_response_to_tweet_id'])
        for _, row in followup_sorted.iterrows():
            followup_map[int(row['in_response_to_tweet_id'])] = {
                "id": int(row['tweet_id']),
                "text": str(row['text']),
                "created_at": str(row['created_at'])
            }
            
    print(f"  Constructed {len(merged):,} initial conversational pairs.")
    
    # Sort chronologically by customer query timestamp
    merged = merged.sort_values('dt_cust').reset_index(drop=True)
    
    # Build clean JSON records
    threads = []
    for idx, row in merged.iterrows():
        b_id = int(row['tweet_id_brand'])
        f_info = followup_map.get(b_id)
        
        # Response latency
        latency_sec = (row['dt_brand'] - row['dt_cust']).total_seconds()
        
        threads.append({
            "thread_id": int(row['tweet_id_cust']),
            "customer_tweet_id": int(row['tweet_id_cust']),
            "customer_text": str(row['text_cust']),
            "customer_created_at": str(row['created_at_cust']),
            "brand_reply_id": b_id,
            "brand_reply_text": str(row['text_brand']),
            "brand_created_at": str(row['created_at_brand']),
            "response_time_seconds": max(0.0, float(latency_sec)),
            "has_followup": f_info is not None,
            "customer_followup_id": f_info["id"] if f_info else None,
            "customer_followup_text": f_info["text"] if f_info else None,
            "customer_followup_created_at": f_info["created_at"] if f_info else None
        })
        
    total_threads = len(threads)
    print(f"\nTotal Reconstructed Threads: {total_threads:,}")
    
    # Temporal Split: 80% historical, 20% held-out
    split_idx = int(total_threads * 0.80)
    historical_threads = threads[:split_idx]
    eval_threads = threads[split_idx:]
    
    print(f"Historical KB / Training Pool: {len(historical_threads):,} threads")
    print(f"  Dates: {historical_threads[0]['customer_created_at']} -> {historical_threads[-1]['customer_created_at']}")
    print(f"Held-out Evaluation Pool:     {len(eval_threads):,} threads")
    print(f"  Dates: {eval_threads[0]['customer_created_at']} -> {eval_threads[-1]['customer_created_at']}")
    
    # Save files
    hist_path = os.path.join(OUTPUT_DIR, "historical_kb.jsonl")
    eval_path = os.path.join(OUTPUT_DIR, "eval_pool.jsonl")
    all_path = os.path.join(OUTPUT_DIR, "apple_threads.jsonl")
    
    print("Writing output files...")
    with open(all_path, "w") as f:
        for t in threads:
            f.write(json.dumps(t) + "\n")
            
    with open(hist_path, "w") as f:
        for t in historical_threads:
            f.write(json.dumps(t) + "\n")
            
    with open(eval_path, "w") as f:
        for t in eval_threads:
            f.write(json.dumps(t) + "\n")
            
    print(f"Successfully generated:")
    print(f"  - {all_path} ({os.path.getsize(all_path)/1024/1024:.1f} MB)")
    print(f"  - {hist_path} ({os.path.getsize(hist_path)/1024/1024:.1f} MB)")
    print(f"  - {eval_path} ({os.path.getsize(eval_path)/1024/1024:.1f} MB)")


if __name__ == "__main__":
    run_thread_reconstruction()
