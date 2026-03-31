import torch
import torchvision.transforms as T
from transformers import CLIPProcessor, CLIPModel
from analysis.probing import RepresentationProber
from data.noise_injector import NoiseInjector
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
import os

class MultimodalProber:
    def __init__(self, model_id="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

    def add_image_noise(self, image_tensor: torch.Tensor, noise_level: float = 0.1) -> torch.Tensor:
        """Adds Gaussian noise to an image tensor."""
        noise = torch.randn_like(image_tensor) * noise_level
        return torch.clamp(image_tensor + noise, 0, 1)

    def analyze_alignment_drift(self, image_url: str, text: str, noise_level: float = 0.2):
        """
        Analyzes how noise in either modality affects the joint embedding space.
        """
        # 1. Prepare raw inputs
        image = Image.open(requests.get(image_url, stream=True).raw)
        
        # 2. Get Clean Embeddings
        inputs = self.processor(text=[text], images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs_clean = self.model(**inputs)
            img_embed_clean = outputs_clean.image_embeds
            txt_embed_clean = outputs_clean.text_embeds

        # 3. Get Noisy Embeddings (Visual Noise)
        noisy_image_tensor = self.add_image_noise(inputs['pixel_values'], noise_level)
        with torch.no_grad():
            img_embed_noisy = self.model.get_image_features(pixel_values=noisy_image_tensor)

        # 4. Get Noisy Embeddings (Text Noise)
        injector = NoiseInjector(self.processor.tokenizer)
        text_ids = self.processor.tokenizer.encode(text)
        noisy_text_ids = injector.random_token_replacement(text_ids, p=noise_level)
        noisy_text = self.processor.tokenizer.decode(noisy_text_ids, skip_special_tokens=True)
        
        inputs_text_noisy = self.processor(text=[noisy_text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            txt_embed_noisy = self.model.get_text_features(**inputs_text_noisy)

        # Compute cosine similarity loss due to noise
        cos = torch.nn.CosineSimilarity(dim=1)
        sim_clean = cos(img_embed_clean, txt_embed_clean).item()
        sim_vis_noise = cos(img_embed_noisy, txt_embed_clean).item()
        sim_txt_noise = cos(img_embed_clean, txt_embed_noisy).item()
        sim_double_noise = cos(img_embed_noisy, txt_embed_noisy).item()

        return {
            "baseline": sim_clean,
            "visual_noise": sim_vis_noise,
            "text_noise": sim_txt_noise,
            "double_noise": sim_double_noise
        }

def run_multimodal_suite():
    # Example: A cat image for testing
    img_url = "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?q=80&w=500&auto=format&fit=crop"
    text = "A photo of a cute kitten sitting on a green field."
    
    prober = MultimodalProber()
    print("\n--- Multimodal Alignment Analysis (CLIP) ---")
    
    noise_levels = [0.0, 0.1, 0.2, 0.4, 0.8]
    results = {"visual": [], "text": [], "double": []}
    
    for p in noise_levels:
        metrics = prober.analyze_alignment_drift(img_url, text, noise_level=p)
        results["visual"].append(metrics["visual_noise"])
        results["text"].append(metrics["text_noise"])
        results["double"].append(metrics["double_noise"])
        print(f"Noise p={p:.1f} | Vis: {metrics['visual_noise']:.3f} | Txt: {metrics['text_noise']:.3f}")

    # Plotting Multimodal Decay Curves
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, results["visual"], 'o-', label="Visual Noise (Gaussian)")
    plt.plot(noise_levels, results["text"], 's-', label="Text Noise (Token Swap)")
    plt.plot(noise_levels, results["double"], '^-', label="Joint Noise (V+T)")
    plt.axhline(y=metrics["baseline"], color='r', linestyle='--', label="Clean Baseline")
    
    plt.xlabel("Noise Level")
    plt.ylabel("Image-Text Cosine Similarity")
    plt.title("The Alignment Breakpoint: Multimodal Noise Sensitivity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if not os.path.exists("experiments"):
        os.makedirs("experiments")
    plt.savefig("experiments/multimodal_alignment_decay.png")
    print("\nMultimodal analysis complete. Results saved to experiments/multimodal_alignment_decay.png")

if __name__ == "__main__":
    run_multimodal_suite()
