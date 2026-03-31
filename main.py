import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from data.loader import get_dataloaders
from analysis.probing import RepresentationProber
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    
    avg_loss = total_loss / (total_tokens / 128) # Approximate batch-normalized loss
    return np.exp(avg_loss)

def evaluate_classification(model_id, dataset_name, noise_type, intensity, device):
    # Load separate classification model for downstream task
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Use DistilBERT for Sentiment140 and AG News
    num_labels = 2 if dataset_name == "sentiment140" else 4
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=num_labels
    ).to(device)
    
    loader = get_dataloaders(tokenizer, noise_type=noise_type, noise_intensity=intensity, dataset_name=dataset_name)
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {dataset_name} ({noise_type} p={intensity})"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids)
            
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            if total > 500: # Limit for demo speed
                break
                
    return correct / total

def run_experiment_suite(dataset_name="ag_news"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "distilgpt2"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    
    prober = RepresentationProber(model, tokenizer)
    
    noise_intensities = [0.0, 0.1, 0.2, 0.3, 0.5]
    results = {"ppl": [], "acc": [], "cka": []}
    
    clean_texts = ["The movie was a masterpiece with great acting.", "Modern AI models are resilient to data noise."]
    clean_states = prober.extract_hidden_states(clean_texts)

    for p in noise_intensities:
        print(f"\n--- Noise Intensity: {p} ---")
        
        # 1. Perplexity Experiment (always on WikiText-2 for representation quality)
        wiki_loader = get_dataloaders(tokenizer, noise_type="token_replacement", noise_intensity=p, dataset_name="wikitext")
        ppl = evaluate_perplexity(model, wiki_loader, device)
        results["ppl"].append(ppl)
        
        # 2. Downstream Task (Classification)
        acc = evaluate_classification(model_id, dataset_name, "token_replacement", p, device)
        results["acc"].append(acc)
        
        # 3. CKA Probing
        noisy_texts = [tokenizer.decode(tokenizer.encode(t, truncation=True, max_length=128), skip_special_tokens=True) for t in clean_texts]
        # (Actually, we need to apply noise manually here or use the injector directly)
        from data.noise_injector import NoiseInjector
        injector = NoiseInjector(tokenizer)
        noisy_ids = [injector.apply_noise(tokenizer.encode(t), noise_type="token_replacement", intensity=p) for t in clean_texts]
        noisy_texts = [tokenizer.decode(ids) for ids in noisy_ids]
        
        noisy_states = prober.extract_hidden_states(noisy_texts)
        cka_sim = prober.compute_cka(clean_states, noisy_states)
        results["cka"].append(cka_sim)
        
        print(f"Results: PPL={ppl:.2f}, ACC={acc:.2f}, CKA={cka_sim:.4f}")

    # Plotting
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(noise_intensities, results["ppl"], marker='o')
    plt.title("Perplexity vs Noise")
    
    plt.subplot(1, 3, 2)
    plt.plot(noise_intensities, results["acc"], marker='s', color='r')
    plt.title("IMDB Accuracy vs Noise")

    plt.subplot(1, 3, 3)
    plt.plot(noise_intensities, results["cka"], marker='^', color='g')
    plt.title("CKA Similarity vs Noise")
    
    plt.tight_layout()
    plt.savefig(f"experiments/{dataset_name}_results_plot.png")
    print(f"\nExperiment complete. Plot saved to experiments/{dataset_name}_results_plot.png")

if __name__ == "__main__":
    import os
    if not os.path.exists("experiments"):
        os.makedirs("experiments")
    
    # Run for AG News by default as requested
    run_experiment_suite(dataset_name="ag_news")
