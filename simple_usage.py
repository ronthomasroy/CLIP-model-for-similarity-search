"""
Simple usage example for the CLIP Similarity Checker
"""

from clip_similarity_checker import CLIPSimilarityChecker


# Initialize the checker
# Threshold: 0.25 is a good starting point
# - Scores above 0.25: Content matches
# - Scores below 0.25: Potential mismatch (fake news indicator)
checker = CLIPSimilarityChecker(threshold=0.25)

# Example: Check if an image matches the text
text = "A cat sitting on a couch in a living room"
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/481px-Cat03.jpg"

result = checker.check_match(text, image_url)

# Access the results
print(f"\nSimilarity Score: {result['similarity_score']}")
print(f"Does it match? {result['is_match']}")
print(f"Confidence: {result['confidence']}")

# Example: Detect fake news (mismatched content)
fake_text = "The new iPhone was announced at Apple's keynote event"
unrelated_image = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Plage_de_Guadeloupe.jpg/500px-Plage_de_Guadeloupe.jpg"

fake_result = checker.check_match(fake_text, unrelated_image)
print(f"\n{'ALERT: Potential Fake News!' if not fake_result['is_match'] else 'Content appears legitimate'}")

# Example: Batch processing multiple pairs
pairs = [
    ("A person playing tennis on a court", "https://example.com/tennis.jpg"),
    ("Breaking news about elections", "https://example.com/news.jpg"),
]

# Uncomment to process multiple pairs:
# results = checker.batch_check(pairs)
