import pandas as pd
import numpy as np
from collections import Counter
import re
import json

data_path = "data/kaggle-dataset/twcs/twcs.csv"

candidate_brands = ["AppleSupport", "SpotifyCares", "AmazonHelp", "Uber_Support"]

print("Filtering dataset for candidates:", candidate_brands)

# We want to collect:
# 1. All outbound tweets from these brands
# 2. All inbound tweets that are replied to by these brands
# 3. Customer followups if any

candidate_outbound = []
candidate_tweet_ids = set()
candidate_inbound_ids = set()

# Read in chunks
chunk_size = 500_000
for chunk in pd.read_csv(data_path, chunksize=chunk_size):
    # Outbound from candidates
    out_mask = chunk['author_id'].isin(candidate_brands)
    out_rows = chunk[out_mask]
    if len(out_rows) > 0:
        candidate_outbound.append(out_rows)
        # Store in_response_to_tweet_id (which are the inbound tweets they replied to)
        valid_in_resp = out_rows['in_response_to_tweet_id'].dropna().astype(int)
        candidate_inbound_ids.update(valid_in_resp.tolist())

out_df = pd.concat(candidate_outbound, ignore_index=True)
print(f"Total candidate outbound replies loaded: {len(out_df):,}")
print(f"Unique inbound tweets to find: {len(candidate_inbound_ids):,}")

# Now find the matching inbound tweets in a second pass
inbound_tweets = []
for chunk in pd.read_csv(data_path, chunksize=chunk_size):
    in_mask = chunk['tweet_id'].isin(candidate_inbound_ids)
    in_rows = chunk[in_mask]
    if len(in_rows) > 0:
        inbound_tweets.append(in_rows)

in_df = pd.concat(inbound_tweets, ignore_index=True)
print(f"Total matching inbound tweets found: {len(in_df):,}")

# Merge inbound with outbound
# inbound tweet_id == outbound in_response_to_tweet_id
pairs = out_df.merge(in_df, left_on='in_response_to_tweet_id', right_on='tweet_id', suffixes=('_brand', '_cust'))

print(f"Total reconstructed customer->brand pairs: {len(pairs):,}")

results = {}

for brand in candidate_brands:
    b_pairs = pairs[pairs['author_id_brand'] == brand]
    n_pairs = len(b_pairs)
    
    # Text characteristics
    avg_cust_len = b_pairs['text_cust'].str.len().mean()
    avg_brand_len = b_pairs['text_brand'].str.len().mean()
    
    # Boilerplate / DM analysis
    dm_count = b_pairs['text_brand'].str.contains(r'\bDM\b|direct message', case=False, regex=True).sum()
    link_count = b_pairs['text_brand'].str.contains(r'http[s]?://', case=False, regex=True).sum()
    question_count = b_pairs['text_brand'].str.contains(r'\?', regex=True).sum()
    
    # Sample 5 representative conversation pairs
    sample_pairs = []
    # Pick a random sample with fixed seed
    samples = b_pairs.sample(n=min(5, n_pairs), random_state=42)
    for _, row in samples.iterrows():
        sample_pairs.append({
            "customer_tweet": row['text_cust'],
            "brand_reply": row['text_brand']
        })
        
    results[brand] = {
        "reply_count": int(n_pairs),
        "avg_cust_chars": float(round(avg_cust_len, 1)),
        "avg_brand_chars": float(round(avg_brand_len, 1)),
        "pct_mentioning_dm": float(round(dm_count / n_pairs * 100, 2)),
        "pct_containing_link": float(round(link_count / n_pairs * 100, 2)),
        "pct_asking_question": float(round(question_count / n_pairs * 100, 2)),
        "samples": sample_pairs
    }
    
    print("\n" + "="*50)
    print(f"BRAND: {brand}")
    print(f"  Reconstructed Pairs: {n_pairs:,}")
    print(f"  Avg Customer Length: {avg_cust_len:.1f} chars")
    print(f"  Avg Brand Length:    {avg_brand_len:.1f} chars")
    print(f"  % Mentioning 'DM':   {dm_count/n_pairs*100:.1f}%")
    print(f"  % Containing Link:   {link_count/n_pairs*100:.1f}%")
    print(f"  % Asking Question:   {question_count/n_pairs*100:.1f}%")
    print("\n  Sample Conversations:")
    for idx, s in enumerate(sample_pairs[:3], 1):
        print(f"  [{idx}] Cust:  {s['customer_tweet']}")
        print(f"      Brand: {s['brand_reply']}")

with open("data/candidate_brands_analysis.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to data/candidate_brands_analysis.json")
