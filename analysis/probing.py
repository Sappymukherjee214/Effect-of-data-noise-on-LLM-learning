import torch
import numpy as np
import umap
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge
import scipy.stats

class RepresentationProber:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    def extract_hidden_states(self, texts: List[str], layer_idx: int = -1) -> torch.Tensor:
        """
        Extract hidden states from a specific layer for multiple texts.
        """
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Hidden states for the specified layer
            hidden_states = outputs.hidden_states[layer_idx]
            # Mean pooling over sentence length
            sentence_embeddings = torch.mean(hidden_states, dim=1)
        return sentence_embeddings

    def visualize_embeddings(self, states_clean: torch.Tensor, states_noisy: torch.Tensor, labels=None):
        """
        Visualizes Clean vs Noisy embeddings using UMAP.
        """
        reducer = umap.UMAP()
        all_states = torch.cat([states_clean, states_noisy], dim=0).cpu().numpy()
        embeddings = reducer.fit_transform(all_states)
        
        n = states_clean.shape[0]
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings[:n, 0], embeddings[:n, 1], c='blue', alpha=0.5, label='Clean')
        plt.scatter(embeddings[n:, 0], embeddings[n:, 1], c='red', alpha=0.5, label='Noisy')
        plt.legend()
        plt.title("UMAP: Change in Embedding Space (Clean vs Noisy)")
        plt.show()

    def compute_cka(self, states1: torch.Tensor, states2: torch.Tensor) -> float:
        """
        Computes Linear Centered Kernel Alignment (CKA) between two representation sets.
        """
        # Ensure they are on CPU for matrix operations if needed, but torch can handle it
        states1 = states1.float()
        states2 = states2.float()
        
        def center(K):
            n = K.shape[0]
            unit = torch.ones([n, n], device=K.device) / n
            return K - unit @ K - K @ unit + unit @ K @ unit

        K = states1 @ states1.T
        L = states2 @ states2.T
        
        Kc = center(K)
        Lc = center(L)
        
        hsic = torch.trace(Kc @ Lc)
        denom = torch.sqrt(torch.trace(Kc @ Kc) * torch.trace(Lc @ Lc))
        
        return (hsic / denom).item()

    def estimate_intrinsic_dimension(self, states: torch.Tensor) -> float:
        """
        Estimates the Intrinsic Dimensionality (ID) of the manifold using the Two-NN algorithm.
        """
        X = states.cpu().numpy()
        
        # Compute distances for the two nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(X)
        distances, _ = nbrs.kneighbors(X)
        
        r1 = distances[:, 1]
        r2 = distances[:, 2]
        
        mask = r1 > 0
        mu = r2[mask] / r1[mask]
        
        mu_sorted = np.sort(mu)
        F_mu = np.arange(1, len(mu_sorted) + 1) / len(mu_sorted)
        
        x = np.log(mu_sorted)
        y = -np.log(1 - F_mu + 1e-10)
        
        slope, _, _, _, _ = scipy.stats.linregress(x, y)
        return slope

    def analyze_noise_propagation(self, texts_clean: List[str], texts_noisy: List[str]) -> Dict[int, float]:
        """
        Analyzes CKA similarity across ALL layers.
        """
        layer_correlations = {}
        device = next(self.model.parameters()).device
        
        inputs_clean = self.tokenizer(texts_clean, return_tensors="pt", padding=True, truncation=True).to(device)
        inputs_noisy = self.tokenizer(texts_noisy, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            out_clean = self.model(**inputs_clean, output_hidden_states=True)
            out_noisy = self.model(**inputs_noisy, output_hidden_states=True)
            
            for i in range(len(out_clean.hidden_states)):
                h_clean = torch.mean(out_clean.hidden_states[i], dim=1)
                h_noisy = torch.mean(out_noisy.hidden_states[i], dim=1)
                sim = self.compute_cka(h_clean, h_noisy)
                layer_correlations[i] = sim
                
        return layer_correlations

    def estimate_mutual_information(self, states: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Estimates Mutual Information I(Z; Y) between representations (Z) and labels (Y).
        """
        from sklearn.feature_selection import mutual_info_classif
        Z = states.cpu().numpy()
        Y = labels.cpu().numpy()
        
        mi_scores = mutual_info_classif(Z, Y, discrete_features=True)
        return np.mean(mi_scores)

    def train_denoising_probe(self, states_noisy: torch.Tensor, states_clean: torch.Tensor):
        """
        Trains a Denoising Probe to 'recover' clean representations from noisy ones.
        """
        X = states_noisy.cpu().numpy()
        Y = states_clean.cpu().numpy()
        
        probe = Ridge(alpha=1.0)
        probe.fit(X, Y)
        
        score = probe.score(X, Y)
        return score, probe

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    
    prober = RepresentationProber(model, tokenizer)
    texts = ["I love this movie!", "Deep learning is amazing."]
    noisy_texts = ["I movie love this!", "Amazing deep is learning."]
    
    clean_states = prober.extract_hidden_states(texts)
    noisy_states = prober.extract_hidden_states(noisy_texts)
    
    print(f"CKA Similarity: {prober.compute_cka(clean_states, noisy_states):.4f}")
