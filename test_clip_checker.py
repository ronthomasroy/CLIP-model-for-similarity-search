"""
Test script for CLIP Similarity Checker
Run this to see the system in action with real examples
"""

from clip_similarity_checker import CLIPSimilarityChecker


def test_fake_news_detection():
    """
    Test the system with various scenarios
    """
    print("="*80)
    print("CLIP-Based Fake News Detection System - Test Suite")
    print("="*80)
    
    # Initialize checker
    checker = CLIPSimilarityChecker(threshold=0.25)
    
    # Test cases
    test_cases = [
        {
            'name': 'Test 1: Matching Content - Cat with Cat Image',
            'text': 'A cute orange and white cat sitting and looking at the camera',
            'image_url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/481px-Cat03.jpg',
            'expected': 'MATCH'
        },
        {
            'name': 'Test 2: Mismatch - Technology Text with Beach Image',
            'text': 'Apple announces new iPhone 15 with revolutionary AI features at their product launch event',
            'image_url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Plage_de_Guadeloupe.jpg/500px-Plage_de_Guadeloupe.jpg',
            'expected': 'MISMATCH'
        },
        {
            'name': 'Test 3: Matching Content - Dog with Dog Image',
            'text': 'A golden retriever dog with beautiful golden fur',
            'image_url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/Golden_Retriever_Dukedestiny01_drvd.jpg/440px-Golden_Retriever_Dukedestiny01_drvd.jpg',
            'expected': 'MATCH'
        },
        {
            'name': 'Test 4: Mismatch - Sports Text with Building Image',
            'text': 'Tennis player serves during championship match at Wimbledon',
            'image_url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/Eiffel_Tower_from_the_Tour_Montparnasse_2014.jpg/400px-Eiffel_Tower_from_the_Tour_Montparnasse_2014.jpg',
            'expected': 'MISMATCH'
        }
    ]
    
    # Track results
    results = []
    
    # Run tests
    for i, test in enumerate(test_cases, 1):
        print(f"\n\n{'='*80}")
        print(f"{test['name']}")
        print(f"Expected: {test['expected']}")
        print('='*80)
        
        result = checker.check_match(
            text=test['text'],
            image_url=test['image_url'],
            verbose=True
        )
        
        if 'error' in result:
            print(f"\n❌ ERROR: {result['error']}")
            results.append({
                'test': test['name'],
                'status': 'ERROR',
                'expected': test['expected'],
                'actual': 'ERROR'
            })
        else:
            actual = 'MATCH' if result['is_match'] else 'MISMATCH'
            status = '✓ PASS' if actual == test['expected'] else '✗ FAIL'
            
            results.append({
                'test': test['name'],
                'status': status,
                'expected': test['expected'],
                'actual': actual,
                'score': result['similarity_score']
            })
    
    # Print summary
    print("\n\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for r in results:
        if r['status'] != 'ERROR':
            print(f"{r['status']:8} | {r['test']}")
            print(f"         | Expected: {r['expected']}, Got: {r['actual']}, Score: {r['score']:.4f}")
        else:
            print(f"{r['status']:8} | {r['test']}")
    
    # Calculate pass rate
    passed = sum(1 for r in results if r['status'] == '✓ PASS')
    total = len(results)
    pass_rate = (passed / total) * 100 if total > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"Pass Rate: {passed}/{total} ({pass_rate:.1f}%)")
    print(f"{'='*80}\n")


def custom_test():
    """
    Run a custom test with your own text and image URL
    """
    print("\n" + "="*80)
    print("CUSTOM TEST")
    print("="*80)
    
    checker = CLIPSimilarityChecker(threshold=0.25)
    
    # Example: Test with your own content
    custom_text = input("\nEnter text description (or press Enter for default): ").strip()
    if not custom_text:
        custom_text = "A person using a laptop computer at a desk"
    
    custom_url = input("Enter image URL (or press Enter for default): ").strip()
    if not custom_url:
        custom_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/481px-Cat03.jpg"
    
    result = checker.check_match(custom_text, custom_url)
    
    if not result['is_match']:
        print("\n⚠️  ALERT: This might be fake news or misleading content!")
        print(f"   The image doesn't match the text (Score: {result['similarity_score']:.4f})")


if __name__ == "__main__":
    import sys
    
    print("\nChoose an option:")
    print("1. Run automated test suite")
    print("2. Run custom test with your own input")
    print("3. Run both")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '1':
        test_fake_news_detection()
    elif choice == '2':
        custom_test()
    elif choice == '3':
        test_fake_news_detection()
        custom_test()
    else:
        print("\nRunning default test suite...")
        test_fake_news_detection()
