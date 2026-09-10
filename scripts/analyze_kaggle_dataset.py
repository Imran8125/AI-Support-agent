import pandas as pd
import numpy as np
from collections import Counter
import re
import json

data_path = "data/kaggle-dataset/twcs/twcs.csv"

print("=" * 60)
print("ANALYZING TWCS DATASET (516 MB)")
print("=" * 60)

total_rows = 0
inbound_count = 0
outbound_count = 0
brands_counter = Counter()
null_in_response = 0
null_response = 0
min_date = None
max_date = None

chunk_size = 250_000

print("Processing in chunks...")
for i, chunk in enumerate(pd.read_csv(data_path, chunksize=chunk_size)):
    total_rows += len(chunk)
    inbound_chunk = chunk[chunk['inbound'] == True]
    outbound_chunk = chunk[chunk['inbound'] == False]
    
    inbound_count += len(inbound_chunk)
    outbound_count += len(outbound_chunk)
    
    # Brands are the authors of outbound tweets
    brands_counter.update(outbound_chunk['author_id'].value_counts().to_dict())
    
    null_in_response += chunk['in_response_to_tweet_id'].isna().sum()
    null_response += chunk['response_tweet_id'].isna().sum()
    
    chunk_dates = pd.to_datetime(chunk['created_at'], errors='coerce')
    chunk_min = chunk_dates.min()
    chunk_max = chunk_dates.max()
    if min_date is None or chunk_min < min_date:
        min_date = chunk_min
    if max_date is None or chunk_max > max_date:
        max_date = chunk_max
    
    print(f"  Processed chunk {i+1}: cumulative {total_rows:,} rows...")

print("\n--- DATASET OVERVIEW ---")
print(f"Total Tweets: {total_rows:,}")
print(f"Inbound Tweets (Customers): {inbound_count:,} ({inbound_count/total_rows*100:.2f}%)")
print(f"Outbound Tweets (Brands): {outbound_count:,} ({outbound_count/total_rows*100:.2f}%)")
print(f"Date Range: {min_date} to {max_date}")
print(f"Tweets without in_response_to_tweet_id (Thread starters / root tweets): {null_in_response:,} ({null_in_response/total_rows*100:.2f}%)")
print(f"Tweets with response_tweet_id: {total_rows - null_response:,} ({(total_rows - null_response)/total_rows*100:.2f}%)")

print("\n--- TOP 20 BRANDS BY OUTBOUND REPLIES ---")
top_brands = brands_counter.most_common(20)
for rank, (brand, count) in enumerate(top_brands, 1):
    print(f"{rank:2d}. {brand:<20} : {count:>8,} replies ({count/outbound_count*100:.2f}%)")

# Save summary to file
summary = {
    "total_tweets": total_rows,
    "inbound_count": inbound_count,
    "outbound_count": outbound_count,
    "date_range": [str(min_date), str(max_date)],
    "top_20_brands": [{"brand": b, "replies": c} for b, c in top_brands]
}

with open("data/twcs_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nSummary saved to data/twcs_summary.json")
