import pandas as pd
import numpy as np
from collections import Counter
import json
import time

data_path = "data/kaggle-dataset/twcs/twcs.csv"

start_time = time.time()
print("Starting fast analysis of TWCS dataset...")

chunk_size = 500_000
total_rows = 0
inbound_count = 0
outbound_count = 0
brands_counter = Counter()
inbound_authors = set()
null_in_response = 0
null_response = 0

sample_dates = []

for i, chunk in enumerate(pd.read_csv(data_path, chunksize=chunk_size)):
    chunk_len = len(chunk)
    total_rows += chunk_len
    
    inbound_mask = chunk['inbound'] == True
    inbound_sub = chunk[inbound_mask]
    outbound_sub = chunk[~inbound_mask]
    
    inbound_count += len(inbound_sub)
    outbound_count += len(outbound_sub)
    
    brands_counter.update(outbound_sub['author_id'].value_counts().to_dict())
    
    null_in_response += chunk['in_response_to_tweet_id'].isna().sum()
    null_response += chunk['response_tweet_id'].isna().sum()
    
    # Collect some dates for bounds
    sample_dates.append(chunk['created_at'].iloc[0])
    sample_dates.append(chunk['created_at'].iloc[-1])
    
    print(f"Processed chunk {i+1}: {total_rows:,} rows elapsed {time.time()-start_time:.1f}s")

top_25_brands = brands_counter.most_common(25)

print("\n" + "="*50)
print(f"TOTAL ROWS: {total_rows:,}")
print(f"INBOUND (Customers): {inbound_count:,} ({inbound_count/total_rows*100:.2f}%)")
print(f"OUTBOUND (Brands): {outbound_count:,} ({outbound_count/total_rows*100:.2f}%)")
print(f"TOTAL UNIQUE BRANDS: {len(brands_counter):,}")
print("="*50)

print("\nTOP 25 BRANDS:")
for rank, (brand, count) in enumerate(top_25_brands, 1):
    print(f"{rank:2d}. {brand:<22} : {count:>8,} replies")

summary = {
    "total_rows": total_rows,
    "inbound_count": int(inbound_count),
    "outbound_count": int(outbound_count),
    "unique_brands_count": len(brands_counter),
    "null_in_response_to": int(null_in_response),
    "null_response_id": int(null_response),
    "sample_dates": sample_dates,
    "top_25_brands": [{"brand": b, "replies": int(c)} for b, c in top_25_brands]
}

with open("data/twcs_top_brands.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nSaved summary to data/twcs_top_brands.json")
