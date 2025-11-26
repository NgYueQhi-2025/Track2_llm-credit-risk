#!/usr/bin/env python3
"""Smoke test for verifying LLM handler and backend app imports and basic functionality."""

import sys
import os

# Set mock mode for testing
os.environ['USE_MOCK_LLM'] = 'true'

def test_llm_handler_import():
    """Test that llm_handler can be imported and has required functions."""
    print("Testing llm_handler import...")
    from llms.backend import llm_handler
    
    # Check for required attributes
    assert hasattr(llm_handler, 'call_llm'), "Missing call_llm function"
    assert hasattr(llm_handler, 'process_text_word_by_word'), "Missing process_text_word_by_word function"
    assert hasattr(llm_handler, '_safe_extract_json'), "Missing _safe_extract_json function"
    
    print("✓ llm_handler import successful")
    print("✓ has call_llm:", hasattr(llm_handler, 'call_llm'))
    print("✓ has process_text_word_by_word:", hasattr(llm_handler, 'process_text_word_by_word'))
    return True


def test_backend_app_import():
    """Test that backend.app can be imported without errors."""
    print("\nTesting backend.app import...")
    from backend import app as demo
    
    # Check for required functions
    assert hasattr(demo, 'process_uploaded_text'), "Missing process_uploaded_text function"
    assert hasattr(demo, 'extract_sentiment'), "Missing extract_sentiment function"
    
    print("✓ backend.app import successful")
    print("✓ has process_uploaded_text:", hasattr(demo, 'process_uploaded_text'))
    print("✓ has extract_sentiment:", hasattr(demo, 'extract_sentiment'))
    return True


def test_process_text_word_by_word_mock():
    """Test process_text_word_by_word with mock data."""
    print("\nTesting process_text_word_by_word with mock data...")
    from llms.backend.llm_handler import process_text_word_by_word
    
    test_text = "I have late payments on multiple loans"
    results = process_text_word_by_word(
        text=test_text,
        mode='extract_risky',
        chunk_size_words=3,
        mock=True
    )
    
    assert isinstance(results, list), "Results should be a list"
    assert len(results) > 0, "Results should not be empty"
    
    # Check structure of first result
    first_result = results[0]
    assert 'chunk' in first_result, "Result missing 'chunk' key"
    assert 'raw' in first_result, "Result missing 'raw' key"
    assert 'parsed' in first_result, "Result missing 'parsed' key"
    assert 'error' in first_result, "Result missing 'error' key"
    
    print(f"✓ Processed {len(results)} chunks")
    print(f"✓ First chunk: {first_result['chunk']}")
    print(f"✓ Parsed result structure validated")
    return True


def test_process_uploaded_text_mock():
    """Test process_uploaded_text helper function with mock data."""
    print("\nTesting process_uploaded_text with mock data...")
    from backend.app import process_uploaded_text
    
    test_text = "I have late payments and missed deadlines on loans"
    results = process_uploaded_text(
        full_text=test_text,
        use_mock=True,
        chunk_size=5
    )
    
    assert isinstance(results, dict), "Results should be a dictionary"
    assert 'risky_phrases' in results, "Missing 'risky_phrases' key"
    assert 'risk_count' in results, "Missing 'risk_count' key"
    assert 'raw_results' in results, "Missing 'raw_results' key"
    
    print(f"✓ Risky phrases: {results['risky_phrases']}")
    print(f"✓ Risk count: {results['risk_count']}")
    print(f"✓ Raw results count: {len(results['raw_results'])}")
    return True


def test_extract_sentiment_mock():
    """Test extract_sentiment helper function with mock data."""
    print("\nTesting extract_sentiment with mock data...")
    from backend.app import extract_sentiment
    
    test_text = "I am confident about my business prospects"
    results = extract_sentiment(
        full_text=test_text,
        use_mock=True,
        chunk_size=3
    )
    
    assert isinstance(results, dict), "Results should be a dictionary"
    assert 'sentiments' in results, "Missing 'sentiments' key"
    assert 'average_score' in results, "Missing 'average_score' key"
    assert 'raw_results' in results, "Missing 'raw_results' key"
    
    print(f"✓ Sentiments: {results['sentiments']}")
    print(f"✓ Average score: {results['average_score']}")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Running Import and Functionality Smoke Tests")
    print("=" * 60)
    
    tests = [
        test_llm_handler_import,
        test_backend_app_import,
        test_process_text_word_by_word_mock,
        test_process_uploaded_text_mock,
        test_extract_sentiment_mock,
    ]
    
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ Test {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    if failed == 0:
        print("All tests passed! ✓")
        print("=" * 60)
        return 0
    else:
        print(f"{failed} test(s) failed ✗")
        print("=" * 60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
