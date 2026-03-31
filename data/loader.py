from datasets import load_dataset
from transformers import AutoTokenizer
from data.noise_injector import NoiseInjector
import torch
import pandas as pd
import kagglehub
import os
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class NoisyTextDataset(Dataset):
    def __init__(self, dataset, tokenizer, noise_injector, noise_type="none", noise_intensity=0.0, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.noise_injector = noise_injector
        self.noise_type = noise_type
        self.noise_intensity = noise_intensity
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text'] if 'text' in item else item['sentence']
        
        # Tokenize
        tokens = self.tokenizer.encode(text, truncation=True, max_length=self.max_length)
        
        # Apply noise
        if self.noise_type != "none":
            tokens = self.noise_injector.apply_noise(tokens, noise_type=self.noise_type, intensity=self.noise_intensity)

        # Pad to max_length
        padded = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        return {
            "input_ids": torch.tensor(padded[:self.max_length]),
            "labels": torch.tensor(padded[:self.max_length]) if 'label' not in item else torch.tensor(item['label'])
        }

def load_sentiment140():
    """Download and load Sentiment140 from Kaggle."""
    path = kagglehub.dataset_download("kazanova/sentiment140")
    # Finding the CSV in the downloaded path
    csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
    full_path = os.path.join(path, csv_file)
    
    # Sentiment140 has no header and uses latin-1 encoding
    df = pd.read_csv(full_path, encoding='latin-1', header=None)
    df.columns = ["label", "id", "date", "flag", "user", "text"]
    # Re-map 4 to 1 for sentiment (0=negative, 4=positive)
    df['label'] = df['label'].map({0: 0, 4: 1})
    return df

def load_ag_news(split="test"):
    """Download and load AG News from Kaggle."""
    path = kagglehub.dataset_download("amananandrai/ag-news-classification-dataset")
    csv_file = "test.csv" if split == "test" else "train.csv"
    full_path = os.path.join(path, csv_file)
    
    df = pd.read_csv(full_path)
    # Combine Title and Description for full context
    df['text'] = df['Title'] + ". " + df['Description']
    # Re-map 1-4 to 0-3
    df['label'] = df['Class Index'] - 1
    return df

def get_dataloaders(tokenizer, batch_size=8, noise_type="none", noise_intensity=0.1, dataset_name="wikitext"):
    """
    Returns DataLoaders for WikiText-2 and IMDB.
    """
    noise_injector = NoiseInjector(tokenizer)
    
    # Load datasets
    if dataset_name == "wikitext":
        raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    elif dataset_name == "sentiment140":
        df = load_sentiment140().sample(5000) # Sample for speed
        raw_dataset = Dataset.from_pandas(df)
    elif dataset_name == "ag_news":
        df = load_ag_news(split="test").sample(2000)
        raw_dataset = Dataset.from_pandas(df)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Prepare datasets
    noisy_dataset = NoisyTextDataset(raw_dataset, tokenizer, noise_injector, noise_type, noise_intensity)
    loader = DataLoader(noisy_dataset, batch_size=batch_size, shuffle=True if dataset_name == "wikitext" else False)

    return loader

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    # Testing AG News
    loader = get_dataloaders(tokenizer, noise_type="token_replacement", noise_intensity=0.2, dataset_name="ag_news")
    
    # Sample check
    for batch in loader:
        print("Sample batch (AG News):", batch['input_ids'][0])
        print("Decoded sample:", tokenizer.decode(batch['input_ids'][0]))
        print("Label:", batch['labels'][0].item())
        break
