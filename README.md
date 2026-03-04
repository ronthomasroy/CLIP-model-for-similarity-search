# CLIP Image-Text Similarity Checker

A tool to detect fake news by comparing image content with textual descriptions using OpenAI's CLIP model.

## Features

- ✅ Download images from URLs
- ✅ Generate embeddings for both text and images
- ✅ Calculate cosine similarity scores
- ✅ Automatic match/mismatch detection
- ✅ Batch processing support
- ✅ Configurable similarity threshold

## Installation

```bash
# Install required packages
pip install -r requirements.txt
```

## Quick Start

```python
from clip_similarity_checker import CLIPSimilarityChecker

# Initialize checker
checker = CLIPSimilarityChecker(threshold=0.25)

# Check if image matches text
text = "Oscar winner holding an award at the ceremony"
image_url = "https://example.com/oscar-photo.jpg"

result = checker.check_match(text, image_url)

print(f"Similarity: {result['similarity_score']}")
print(f"Match: {result['is_match']}")
```

## How It Works

1. **Download Image**: Fetches the image from the provided URL
2. **Encode**: Uses CLIP to encode both text and image into the same 512-dimensional embedding space
3. **Compare**: Calculates cosine similarity between the embeddings
4. **Decide**: Compares similarity score against threshold to determine match/mismatch

## Understanding Similarity Scores

- **0.40 - 1.00**: Strong match (content clearly relates)
- **0.25 - 0.40**: Moderate match (some relation)
- **0.00 - 0.25**: Weak/no match (potential fake news)

## Threshold Configuration

The default threshold is `0.25`. You can adjust it based on your needs:

```python
# Stricter matching (fewer false positives)
checker = CLIPSimilarityChecker(threshold=0.35)

# More lenient matching (fewer false negatives)
checker = CLIPSimilarityChecker(threshold=0.20)
```

## Result Structure

```python
{
    'similarity_score': 0.3421,      # Cosine similarity (0-1)
    'is_match': True,                # Whether it passes threshold
    'threshold': 0.25,               # The threshold used
    'confidence': 'HIGH',            # HIGH or LOW confidence
    'text': '...',                   # Original text
    'image_url': '...'              # Original image URL
}
```

## Use Cases

### 1. Fake News Detection
Identify articles where images don't match the content:
```python
text = "President gives speech at climate summit"
image_url = "https://example.com/random-beach.jpg"
result = checker.check_match(text, image_url)

if not result['is_match']:
    print("⚠️ Potential fake news detected!")
```

### 2. Content Verification
Verify social media posts:
```python
tweet_text = "My new puppy playing in the garden"
tweet_image = "https://example.com/image.jpg"
result = checker.check_match(tweet_text, tweet_image)
```

### 3. Batch Processing
Check multiple articles:
```python
articles = [
    ("Article 1 text...", "https://example.com/img1.jpg"),
    ("Article 2 text...", "https://example.com/img2.jpg"),
]

results = checker.batch_check(articles)
```

## Limitations

- **Metaphorical Images**: News articles often use symbolic/stock photos that may not literally match
- **Abstract Concepts**: CLIP may struggle with very abstract relationships
- **Context**: Single image-text pair doesn't capture full article context
- **Legitimate Variety**: Some real news uses artistic/illustrative images

## Best Practices

1. **Use as one signal**: Combine with other fake news detection methods
2. **Consider context**: A mismatch doesn't always mean fake news
3. **Adjust threshold**: Fine-tune based on your specific use case
4. **Human review**: Use for flagging, not automatic removal

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- PIL (Pillow)
- Requests

## Performance

- First run downloads the CLIP model (~600MB)
- Subsequent runs are fast (1-2 seconds per check)
- GPU support for faster processing

## Troubleshooting

**Issue**: Image download fails
- **Solution**: Check URL accessibility and format

**Issue**: Low similarity for matching content
- **Solution**: Lower the threshold or provide more descriptive text

**Issue**: High similarity for mismatched content
- **Solution**: Increase threshold or use more specific text
