import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from data.loader import get_dataloaders
from analysis.probing import RepresentationProber
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def evaluate_perplexity(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Perplexity"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, labels=labels)
            
            loss = outputs.loss
            total_loss += loss.item() * input_ids.size(0)
            total_tokens += input_ids.size(1) * input_ids.size(0)
    
    avg_loss = total_loss / (total_tokens / 128) 
    return np.exp(avg_loss)

def run_noise_comparison():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "distilgpt2"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    prober = RepresentationProber(model, tokenizer)
    
    results = {}

    # 1. Baseline: AG News Clean
    print("\n--- Baseline: AG News (Clean) ---")
    ag_clean_loader = get_dataloaders(tokenizer, noise_type="none", dataset_name="ag_news")
    ppl_clean = evaluate_perplexity(model, ag_clean_loader, device)
    
    # 2. Synthetic Noise: AG News + Token Replacement (p=0.15 to match high natural noise)
    print("\n--- Case A: Synthetic Noise (AG News p=0.15) ---")
    ag_noisy_loader = get_dataloaders(tokenizer, noise_type="token_replacement", noise_intensity=0.15, dataset_name="ag_news")
    ppl_synthetic = evaluate_perplexity(model, ag_noisy_loader, device)
    
    # 3. Natural Noise: Sentiment140 (Twitter handles, slang, typos)
    print("\n--- Case B: Natural Noise (Sentiment140 Twitter) ---")
    s140_loader = get_dataloaders(tokenizer, noise_type="none", dataset_name="sentiment140")
    ppl_natural = evaluate_perplexity(model, s140_loader, device)
    
    # Representation Distance (CKA Similarity to AG News Clean)
    # We take a sample of 100 sentences for CKA
    ag_clean_texts = [tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True) for batch in ag_clean_loader][0:100]
    ag_noisy_texts = [tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True) for batch in ag_noisy_loader][0:100]
    s140_texts = [tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True) for batch in s140_loader][0:100]
    
    states_clean = prober.extract_hidden_states(ag_clean_texts)
    states_synthetic = prober.extract_hidden_states(ag_noisy_texts)
    states_natural = prober.extract_hidden_states(s140_texts)
    
    cka_synthetic = prober.compute_cka(states_clean, states_synthetic)
    cka_natural = prober.compute_cka(states_clean, states_natural)
    
    print("\n" + "="*40)
    print("NOISE COMPARISON RESULTS")
    print("="*40)
    print(f"AG News Clean PPL:    {ppl_clean:.4f}")
    print(f"Synthetic Noise PPL:  {ppl_synthetic:.4f} (CKA Similarity: {cka_synthetic:.4f})")
    print(f"Natural Noise PPL:    {ppl_natural:.4f} (CKA Similarity: {cka_natural:.4f})")
    print("="*40)

    # Visualization
    labels = ['Clean', 'Synthetic (0.15)', 'Natural (S140)']
    ppls = [ppl_clean, ppl_synthetic, ppl_natural]
    ckas = [1.0, cka_synthetic, cka_natural]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(labels, ppls, color=['blue', 'red', 'orange'])
    plt.title("Perplexity Comparison (Lower is Better)")
    plt.ylabel("PPL")

    plt.subplot(1, 2, 2)
    plt.bar(labels, ckas, color=['blue', 'red', 'orange'])
    plt.title("CKA Similarity to Clean (Higher is Better)")
    plt.ylabel("CKA Similarity")
    
    if not os.path.exists("experiments"):
        os.makedirs("experiments")
    plt.savefig("experiments/noise_comparison.png")
    print("\nComparison plot saved to experiments/noise_comparison.png")

if __name__ == "__main__":
    run_noise_comparison()
