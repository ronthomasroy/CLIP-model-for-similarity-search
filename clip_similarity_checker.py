"""
CLIP-based Image-Text Similarity Checker
Compares image content with textual descriptions to detect mismatches
"""

import torch
import requests
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
import numpy as np


class CLIPSimilarityChecker:
    """
    A class to check similarity between images and text using CLIP model
    """
    
    def __init__(self, model_name="openai/clip-vit-base-patch32", threshold=0.25):
        """
        Initialize the CLIP model and processor
        
        Args:
            model_name (str): The CLIP model to use
            threshold (float): Similarity threshold for matching (0-1)
        """
        print(f"Loading CLIP model: {model_name}...")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.threshold = threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model loaded successfully on {self.device}")
        
    def download_image(self, image_url):
        """
        Download image from URL
        
        Args:
            image_url (str): URL of the image
            
        Returns:
            PIL.Image: Downloaded image
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(image_url, headers=headers, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return image
        except Exception as e:
            raise Exception(f"Failed to download image from {image_url}: {str(e)}")
    
    def get_embeddings(self, text, image):
        """
        Get CLIP embeddings for text and image
        
        Args:
            text (str): Text content to encode
            image (PIL.Image): Image to encode
            
        Returns:
            tuple: (text_embedding, image_embedding)
        """
        # Process inputs
        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            text_embeds = outputs.text_embeds
            image_embeds = outputs.image_embeds
            
        # Normalize embeddings
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        
        return text_embeds, image_embeds
    
    def calculate_similarity(self, text_embeds, image_embeds):
        """
        Calculate cosine similarity between text and image embeddings
        
        Args:
            text_embeds (torch.Tensor): Text embeddings
            image_embeds (torch.Tensor): Image embeddings
            
        Returns:
            float: Cosine similarity score (0-1)
        """
        similarity = torch.cosine_similarity(text_embeds, image_embeds)
        return similarity.item()
    
    def check_match(self, text, image_url, verbose=True):
        """
        Check if image and text content match
        
        Args:
            text (str): Text content
            image_url (str): URL of the image
            verbose (bool): Print detailed information
            
        Returns:
            dict: Results containing similarity score and match status
        """
        try:
            # Download image
            if verbose:
                print(f"\nDownloading image from: {image_url}")
            image = self.download_image(image_url)
            
            if verbose:
                print(f"Image size: {image.size}")
                print(f"Text preview: {text[:100]}...")
            
            # Get embeddings
            if verbose:
                print("Generating embeddings...")
            text_embeds, image_embeds = self.get_embeddings(text, image)
            
            # Calculate similarity
            similarity_score = self.calculate_similarity(text_embeds, image_embeds)
            
            # Determine if it's a match
            is_match = similarity_score >= self.threshold
            
            result = {
                'similarity_score': round(similarity_score, 4),
                'is_match': is_match,
                'threshold': self.threshold,
                'text': text,
                'image_url': image_url,
                'confidence': 'HIGH' if abs(similarity_score - self.threshold) > 0.1 else 'LOW'
            }
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"Similarity Score: {result['similarity_score']:.4f}")
                print(f"Threshold: {result['threshold']:.4f}")
                print(f"Match: {'✓ YES' if is_match else '✗ NO'}")
                print(f"Confidence: {result['confidence']}")
                print(f"{'='*60}")
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'similarity_score': None,
                'is_match': None
            }
    
    def batch_check(self, text_image_pairs):
        """
        Check multiple text-image pairs
        
        Args:
            text_image_pairs (list): List of (text, image_url) tuples
            
        Returns:
            list: List of result dictionaries
        """
        results = []
        for i, (text, image_url) in enumerate(text_image_pairs, 1):
            print(f"\n\nProcessing pair {i}/{len(text_image_pairs)}")
            result = self.check_match(text, image_url, verbose=True)
            results.append(result)
        
        return results


def main():
    """
    Example usage of the CLIP similarity checker
    """
    # Initialize checker with threshold
    checker = CLIPSimilarityChecker(threshold=0.25)
    
    # Example 1: Matching content (Oscar winner with Oscar ceremony)
    print("\n" + "="*60)
    print("EXAMPLE 1: Matching Content")
    print("="*60)
    
    text1 = "Emma Stone won the Academy Award for Best Actress for her role in Poor Things at the 2024 Oscars ceremony"
    image_url1 = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/Emma_Stone_at_the_39th_G%C3%B6teborg_International_Film_Festival_%28cropped%29.jpg/440px-Emma_Stone_at_the_39th_G%C3%B6teborg_International_Film_Festival_%28cropped%29.jpg"
    
    result1 = checker.check_match(text1, image_url1)
    
    # Example 2: Mismatching content (Oscar text with beach image)
    print("\n\n" + "="*60)
    print("EXAMPLE 2: Mismatching Content (Potential Fake News)")
    print("="*60)
    
    text2 = "Leonardo DiCaprio won the Oscar for Best Actor at the Academy Awards ceremony"
    image_url2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Plage_de_Guadeloupe.jpg/500px-Plage_de_Guadeloupe.jpg"
    
    result2 = checker.check_match(text2, image_url2)
    
    # Example 3: Another matching example
    print("\n\n" + "="*60)
    print("EXAMPLE 3: Dog Description with Dog Image")
    print("="*60)
    
    text3 = "A golden retriever dog playing in the park on a sunny day"
    image_url3 = "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/Golden_Retriever_Dukedestiny01_drvd.jpg/440px-Golden_Retriever_Dukedestiny01_drvd.jpg"
    
    result3 = checker.check_match(text3, image_url3)
    
    # Summary
    print("\n\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    examples = [
        ("Oscar winner with portrait", result1),
        ("Oscar text with beach image", result2),
        ("Dog text with dog image", result3)
    ]
    
    for desc, result in examples:
        if 'error' not in result:
            status = "MATCH ✓" if result['is_match'] else "MISMATCH ✗"
            print(f"{desc:.<50} {status} (Score: {result['similarity_score']:.4f})")


if __name__ == "__main__":
    main()
