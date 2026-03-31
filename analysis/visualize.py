import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from analysis.probing import RepresentationProber
from data.noise_injector import NoiseInjector
import matplotlib.pyplot as plt
import os

def visualize_noise_impact(model_id="distilgpt2"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    
    prober = RepresentationProber(model, tokenizer)
    injector = NoiseInjector(tokenizer)

    # Sample texts for visualization
    texts_clean = [
        "The quick brown fox jumps over the lazy dog.",
        "A beautiful sunshine day brings joy to everyone.",
        "Scientific research thrives on robust data analysis.",
        "Natural language processing is a subset of machine learning.",
        "Consistency is the key to mastering any skill.",
        "Heavy rain caused flooding in the low-lying areas.",
        "The economy is showing signs of steady recovery.",
        "Exploring deep space is a dream for many nations.",
        "Public health policies save millions of lives yearly.",
        "Art and music express the depth of human emotions."
    ]

    noise_intensities = [0.1, 0.3, 0.5]
    
    # Pre-tokenize all for consistent processing
    clean_ids = [tokenizer.encode(t, truncation=True, max_length=128) for t in texts_clean]
    clean_states = prober.extract_hidden_states(texts_clean)
    
    plt.figure(figsize=(18, 5))
    
    for idx, p in enumerate(noise_intensities):
        noisy_ids = [injector.apply_noise(ids, noise_type="token_replacement", intensity=p) for ids in clean_ids]
        noisy_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in noisy_ids]
        
        noisy_states = prober.extract_hidden_states(noisy_texts)
        
        # Plotting logic integrated here
        import umap
        reducer = umap.UMAP(n_neighbors=5, min_dist=0.3)
        all_states = torch.cat([clean_states, noisy_states], dim=0).cpu().numpy()
        embeddings = reducer.fit_transform(all_states)
        
        n = clean_states.shape[0]
        plt.subplot(1, 3, idx + 1)
        plt.scatter(embeddings[:n, 0], embeddings[:n, 1], c='blue', alpha=0.7, label='Clean', edgecolor='k')
        plt.scatter(embeddings[n:, 0], embeddings[n:, 1], c='red', alpha=0.7, label=f'Noisy (p={p})', edgecolor='k')
        
        # Connect identical pairs to show drift
        for i in range(n):
            plt.plot([embeddings[i, 0], embeddings[n + i, 0]], [embeddings[i, 1], embeddings[n + i, 1]], 'k--', alpha=0.2)
            
        plt.title(f"Representation Displacement (p={p})")
        plt.legend()
        
    if not os.path.exists("experiments"):
        os.makedirs("experiments")
        
    plt.tight_layout()
    plt.savefig("experiments/representation_displacement.png")
    print("UMAP plot saved to experiments/representation_displacement.png")

if __name__ == "__main__":
    visualize_noise_impact()
