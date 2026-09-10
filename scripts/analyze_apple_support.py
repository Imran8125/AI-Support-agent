import pandas as pd
import numpy as np
import re
import json

data_path = "data/kaggle-dataset/twcs/twcs.csv"

print("Deep dive on AppleSupport...")

# Find all tweets related to AppleSupport:
# 1. outbound from AppleSupport
# 2. inbound to AppleSupport
# We can read chunks and collect them.

apple_out = []
apple_in_ids = set()

chunk_size = 500_000
for chunk in pd.read_csv(data_path, chunksize=chunk_size):
    out_mask = chunk['author_id'] == 'AppleSupport'
    out_rows = chunk[out_mask]
    if len(out_rows) > 0:
        apple_out.append(out_rows)
        valid_in = out_rows['in_response_to_tweet_id'].dropna().astype(int)
        apple_in_ids.update(valid_in.tolist())

out_df = pd.concat(apple_out, ignore_index=True)
print(f"AppleSupport outbound tweets: {len(out_df):,}")

# Inbound
apple_in = []
for chunk in pd.read_csv(data_path, chunksize=chunk_size):
    in_mask = chunk['tweet_id'].isin(apple_in_ids)
    in_rows = chunk[in_mask]
    if len(in_rows) > 0:
        apple_in.append(in_rows)

in_df = pd.concat(apple_in, ignore_index=True)
print(f"Matching customer inbound tweets: {len(in_df):,}")

# Merge pairs
pairs = out_df.merge(in_df, left_on='in_response_to_tweet_id', right_on='tweet_id', suffixes=('_brand', '_cust'))

# Check English proportion (heuristic: ascii chars ratio > 0.85)
def is_likely_english(text):
    if not isinstance(text, str) or not text.strip():
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return (ascii_chars / len(text)) > 0.85

eng_cust = pairs['text_cust'].apply(is_likely_english)
print(f"English customer tweets percentage: {eng_cust.mean()*100:.2f}%")

# Thread followups: does customer reply back after brand reply?
# in out_df, response_tweet_id contains tweet ids that responded to brand
has_cust_followup = pairs['response_tweet_id_brand'].notna() & (pairs['response_tweet_id_brand'] != '')
print(f"Pairs where customer followed up after brand reply: {has_cust_followup.sum():,} ({has_cust_followup.mean()*100:.2f}%)")
print(f"Pairs where customer did NOT follow up (single-reply resolution): {(~has_cust_followup).sum():,} ({(~has_cust_followup).mean()*100:.2f}%)")

# Sample 10 customer queries to see common topics
print("\n--- SAMPLE 10 CUSTOMER QUERIES TO APPLESUPPORT ---")
for i, txt in enumerate(pairs['text_cust'].sample(10, random_state=123)):
    clean_txt = re.sub(r'@\w+', '', txt).strip()
    clean_txt = re.sub(r'https?://\S+', '', clean_txt).strip()
    print(f"{i+1:2d}. {clean_txt}")

