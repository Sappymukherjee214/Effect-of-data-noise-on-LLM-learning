import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from data.loader import get_dataloaders
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class UncertaintyEstimator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    def compute_ece(self, n_bins=10):
        """
        Computes Expected Calibration Error (ECE).
        How well does the model's 'belief' match reality?
        """
        # Placeholder for collected predictions
        all_probs = []
        all_labels = []
        all_preds = []

        # Implementation Logic:
        # Bin probabilities into n_bins [0, 1].
        # For each bin, compute avg accuracy and avg confidence.
        # ECE = sum( |accuracy_bin - confidence_bin| * bin_weight )
        pass

    def mc_dropout_inference(self, text: str, n_samples: int = 10):
        """
        Performs Monte Carlo Dropout to estimate model uncertainty (Epistemic).
        """
        self.model.train() # Enable dropout
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        logits_list = []
        with torch.no_grad():
            for _ in range(n_samples):
                outputs = self.model(**inputs)
                logits_list.append(F.softmax(outputs.logits, dim=1))
        
        all_probs = torch.stack(logits_list) # [n_samples, 1, num_labels]
        mean_prob = torch.mean(all_probs, dim=0)
        variance = torch.var(all_probs, dim=0) # Measure of disagreement
        
        entropy = -torch.sum(mean_prob * torch.log(mean_prob + 1e-10), dim=1)
        return mean_prob, variance, entropy

def run_uncertainty_research(dataset_name="ag_news"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=4
    ).to(device)
    
    estimator = UncertaintyEstimator(model, tokenizer)
    
    noise_intensities = [0.0, 0.2, 0.5]
    results = {"entropy": [], "variance": []}
    
    sample_text = "The launch of the newest satellite into orbit marks a significant milestone for SpaceX."
    
    print("\n--- Uncertainty Growth Under Noise ---")
    for p in noise_intensities:
        # Inject noise into text
        # (Using specific injector to keep it focused)
        from data.noise_injector import NoiseInjector
        injector = NoiseInjector(tokenizer)
        text_ids = tokenizer.encode(sample_text)
        noisy_ids = injector.apply_noise(text_ids, intensity=p)
        noisy_text = tokenizer.decode(noisy_ids, skip_special_tokens=True)
        
        mean_p, var, ent = estimator.mc_dropout_inference(noisy_text)
        results["entropy"].append(ent.item())
        results["variance"].append(torch.mean(var).item())
        
        print(f"Noise p={p:.1f} | Predictive Entropy: {ent.item():.4f} | Predictive Variance: {torch.mean(var).item():.6f}")

    # Plotting Entropy-Noise relationship
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(noise_intensities, results["entropy"], 'r-o', label="Entropy")
    plt.xlabel("Noise Level")
    plt.title("Information Entropy (Confused Prediction)")
    
    plt.subplot(1, 2, 2)
    plt.plot(noise_intensities, results["variance"], 'g-s', label="MC Variance")
    plt.xlabel("Noise Level")
    plt.title("Epistemic Uncertainty (MC Dropout)")
    
    plt.tight_layout()
    if not os.path.exists("experiments"):
        os.makedirs("experiments")
    plt.savefig("experiments/uncertainty_scaling.png")
    print("\nUncertainty analysis complete. Saved to experiments/uncertainty_scaling.png")

if __name__ == "__main__":
    run_uncertainty_research()
