import pandas as pd
import json

train_df = pd.read_csv("data/banking77/train.csv")
test_df = pd.read_csv("data/banking77/test.csv")
with open("data/banking77/categories.json") as f:
    categories = json.load(f)

print("=" * 50)
print("BANKING77 DATASET ANALYSIS")
print("=" * 50)
print(f"Total training queries: {len(train_df):,}")
print(f"Total test queries:     {len(test_df):,}")
print(f"Combined total:         {len(train_df) + len(test_df):,}")
print(f"Number of categories:   {len(categories)}")

# Check class balance
train_counts = train_df['category'].value_counts()
print(f"\nTrain min class count: {train_counts.min()}, max: {train_counts.max()}, median: {train_counts.median()}")
test_counts = test_df['category'].value_counts()
print(f"Test min class count: {test_counts.min()}, max: {test_counts.max()}, median: {test_counts.median()}")

# Query lengths
train_df['char_len'] = train_df['text'].str.len()
train_df['word_len'] = train_df['text'].str.split().str.len()
print(f"\nAverage character length: {train_df['char_len'].mean():.1f} (min: {train_df['char_len'].min()}, max: {train_df['char_len'].max()})")
print(f"Average word length:      {train_df['word_len'].mean():.1f} (min: {train_df['word_len'].min()}, max: {train_df['word_len'].max()})")

# Sample categories and queries
print("\nSample Categories & Queries:")
sample_cats = ["card_arrival", "lost_or_stolen_card", "verify_source_of_funds", "exchange_rate", "unable_to_verify_identity"]
for cat in sample_cats:
    sample_text = train_df[train_df['category'] == cat]['text'].iloc[0]
    print(f" - [{cat}]: \"{sample_text}\"")

# Key differences between Banking77 and Kaggle Twitter
print("\nKey Comparison: Banking77 vs. Kaggle Twitter (TWCS):")
print("1. Structure: Banking77 is curated, single-turn sentence queries with pre-labeled ground truth (77 fine-grained classes).")
print("2. Twitter TWCS is uncurated, multi-turn, conversational, containing emojis, URLs, typos, brand mentions, and has NO pre-existing intent labels.")
print("3. Relevance to assignment: Banking77 can serve as a benchmark / validation for intent classification algorithms, but the primary task requires deriving intents directly from Twitter data.")

