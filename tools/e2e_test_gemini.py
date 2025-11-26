#!/usr/bin/env python
"""
End-to-End Test for Gemini RAG Integration

Tests the complete RAG pipeline:
1. Load or build RAG index
2. Process sample examples through call_llm_with_rag
3. Display parsed outputs including risky_phrases, summary, and CRF fields
4. Works in both mock mode (no API key) and real Gemini mode

Usage:
    # Using sample examples (default)
    python tools/e2e_test_gemini.py
    
    # Using custom examples file
    python tools/e2e_test_gemini.py --examples tools/sample_examples.json
    
    # Force mock mode (no API calls)
    python tools/e2e_test_gemini.py --mock
    
    # Real Gemini mode (requires GOOGLE_API_KEY)
    python tools/e2e_test_gemini.py --no-mock
    
    # Build index first if missing
    python tools/e2e_test_gemini.py --build-index

Environment Variables:
    GOOGLE_API_KEY or GEMINI_API_KEY: Required for real Gemini mode
    GEMINI_MODEL: Generation model (default: gemini-2.0-flash-exp)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from llms.backend import llm_handler
    from llms.backend import gemini_adapter
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you're running from the repository root.")
    sys.exit(1)


def load_examples(examples_path: str) -> List[Dict]:
    """Load test examples from JSON file."""
    try:
        with open(examples_path, 'r', encoding='utf-8') as f:
            examples = json.load(f)
        print(f"✓ Loaded {len(examples)} examples from {examples_path}")
        return examples
    except Exception as e:
        print(f"Error loading examples: {e}")
        return []


def build_index_if_missing(index_dir: str, force: bool = False):
    """Build RAG index if it doesn't exist."""
    docs_path = os.path.join(index_dir, 'docs.jsonl')
    embeddings_path = os.path.join(index_dir, 'embeddings.npy')
    
    if not force and os.path.exists(docs_path) and os.path.exists(embeddings_path):
        print(f"✓ RAG index exists in {index_dir}")
        return True
    
    print(f"Building RAG index in {index_dir}...")
    try:
        # Import and run build script
        from tools import build_rag_index_gemini
        success = build_rag_index_gemini.build_index(
            input_path=None,
            output_dir=index_dir,
            provider='gemini' if gemini_adapter.is_gemini_available() else 'local',
            generate_sample=True
        )
        if success:
            print(f"✓ RAG index built successfully")
        return success
    except Exception as e:
        print(f"Failed to build index: {e}")
        return False


def format_result(result: Dict, example_name: str) -> str:
    """Format test result for display."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"TEST: {example_name}")
    lines.append("=" * 80)
    
    # Summary
    summary = result.get('summary', 'N/A')
    lines.append(f"Summary: {summary}")
    
    # Risk assessment
    risk_level = result.get('risk_level', 'N/A')
    risk_score = result.get('risk_score', 'N/A')
    confidence = result.get('confidence', 'N/A')
    lines.append(f"Risk Level: {risk_level}")
    lines.append(f"Risk Score: {risk_score}")
    lines.append(f"Confidence: {confidence}")
    
    # Risky phrases
    risky_phrases = result.get('risky_phrases', [])
    if risky_phrases:
        lines.append(f"Risky Phrases: {', '.join(risky_phrases)}")
    else:
        lines.append("Risky Phrases: None detected")
    
    # Mitigating factors
    mitigating = result.get('mitigating_factors', [])
    if mitigating:
        lines.append(f"Mitigating Factors: {', '.join(mitigating)}")
    
    # Risk factors
    risk_factors = result.get('risk_factors', [])
    if risk_factors:
        lines.append(f"Risk Factors: {', '.join(risk_factors)}")
    
    # Negation detection
    negation = result.get('negation_detected', 'N/A')
    lines.append(f"Negation Detected: {negation}")
    
    # Temporal factors
    temporal = result.get('temporal_factors', [])
    if temporal:
        lines.append(f"Temporal Factors: {', '.join(temporal)}")
    
    # Error if present
    error = result.get('error')
    if error:
        lines.append(f"ERROR: {error}")
    
    lines.append("")
    return "\n".join(lines)


def run_e2e_test(
    examples_path: str = 'tools/sample_examples.json',
    index_dir: str = 'backend/llm_index',
    mock: bool = None,
    build_index: bool = False,
    top_k: int = 3
):
    """Run end-to-end test of RAG pipeline."""
    
    print("=" * 80)
    print("Gemini RAG End-to-End Test")
    print("=" * 80)
    print()
    
    # Check environment
    has_gemini = gemini_adapter.is_gemini_available() if hasattr(gemini_adapter, 'is_gemini_available') else False
    has_local = gemini_adapter.is_local_embedder_available() if hasattr(gemini_adapter, 'is_local_embedder_available') else False
    
    print(f"Gemini available: {has_gemini}")
    print(f"Local embedder available: {has_local}")
    
    # Determine mock mode
    if mock is None:
        # Auto-detect: use mock if Gemini not available
        mock = not has_gemini
        print(f"Auto-detected mode: {'MOCK' if mock else 'REAL GEMINI'}")
    else:
        print(f"Mode: {'MOCK' if mock else 'REAL GEMINI'}")
    
    if not mock and not has_gemini:
        print("\nWARNING: Real mode requested but Gemini not available.")
        print("Please set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
        print("Falling back to MOCK mode.")
        mock = True
    
    print()
    
    # Build index if requested or missing
    if build_index or not mock:
        if not build_index_if_missing(index_dir, force=build_index):
            print("WARNING: Failed to build index, continuing anyway...")
    
    print()
    
    # Load examples
    examples = load_examples(examples_path)
    if not examples:
        print("No examples to test. Using default example.")
        examples = [{
            "id": 1,
            "name": "Default Test",
            "text": "I have no missed payments and always pay on time."
        }]
    
    print(f"\nRunning {len(examples)} tests...")
    print()
    
    # Run tests
    results = []
    for i, example in enumerate(examples, 1):
        example_id = example.get('id', i)
        name = example.get('name', f'Example {example_id}')
        text = example.get('text', '')
        
        if not text:
            print(f"Skipping example {example_id}: no text")
            continue
        
        print(f"[{i}/{len(examples)}] Testing: {name}")
        print(f"Input text: {text[:100]}{'...' if len(text) > 100 else ''}")
        
        try:
            # Call RAG pipeline
            result = llm_handler.call_llm_with_rag(
                text=text,
                index_dir=index_dir,
                top_k=top_k,
                use_cache=True,
                mock=mock
            )
            
            results.append({
                'example': example,
                'result': result,
                'success': True
            })
            
            print(f"✓ Success: {result.get('risk_level', 'N/A')} risk")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            results.append({
                'example': example,
                'result': {'error': str(e)},
                'success': False
            })
        
        print()
    
    # Display detailed results
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)
    print()
    
    for item in results:
        example = item['example']
        result = item['result']
        name = example.get('name', f"Example {example.get('id', '?')}")
        
        output = format_result(result, name)
        print(output)
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    
    print("=" * 80)
    print(f"SUMMARY: {successful}/{total} tests completed successfully")
    print("=" * 80)
    
    # Categorize by risk level
    risk_counts = {'low': 0, 'medium': 0, 'high': 0, 'error': 0}
    for item in results:
        if item['success']:
            risk_level = item['result'].get('risk_level', 'unknown').lower()
            if risk_level in risk_counts:
                risk_counts[risk_level] += 1
        else:
            risk_counts['error'] += 1
    
    print(f"\nRisk distribution:")
    print(f"  Low risk:    {risk_counts['low']}")
    print(f"  Medium risk: {risk_counts['medium']}")
    print(f"  High risk:   {risk_counts['high']}")
    if risk_counts['error'] > 0:
        print(f"  Errors:      {risk_counts['error']}")
    
    return successful == total


def main():
    parser = argparse.ArgumentParser(
        description='End-to-end test for Gemini RAG integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default sample examples in mock mode
  python tools/e2e_test_gemini.py
  
  # Run with custom examples in real Gemini mode
  python tools/e2e_test_gemini.py --examples my_examples.json --no-mock
  
  # Build index and run tests
  python tools/e2e_test_gemini.py --build-index
  
  # Force mock mode even with API key set
  python tools/e2e_test_gemini.py --mock
        """
    )
    
    parser.add_argument(
        '--examples', '-e',
        type=str,
        default='tools/sample_examples.json',
        help='Path to JSON file with test examples (default: tools/sample_examples.json)'
    )
    parser.add_argument(
        '--index-dir', '-i',
        type=str,
        default='backend/llm_index',
        help='Directory containing RAG index (default: backend/llm_index)'
    )
    parser.add_argument(
        '--mock', '-m',
        action='store_true',
        help='Force mock mode (no API calls)'
    )
    parser.add_argument(
        '--no-mock',
        action='store_true',
        help='Force real Gemini mode (requires API key)'
    )
    parser.add_argument(
        '--build-index', '-b',
        action='store_true',
        help='Build/rebuild RAG index before testing'
    )
    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=3,
        help='Number of relevant chunks to retrieve (default: 3)'
    )
    
    args = parser.parse_args()
    
    # Determine mock mode
    if args.mock and args.no_mock:
        parser.error("Cannot specify both --mock and --no-mock")
    
    mock = None
    if args.mock:
        mock = True
    elif args.no_mock:
        mock = False
    
    # Run test
    success = run_e2e_test(
        examples_path=args.examples,
        index_dir=args.index_dir,
        mock=mock,
        build_index=args.build_index,
        top_k=args.top_k
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
