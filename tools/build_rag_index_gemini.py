#!/usr/bin/env python
"""
Build RAG Index for Gemini Integration

This script builds a RAG (Retrieval-Augmented Generation) index by:
1. Loading or generating credit risk research documents
2. Chunking documents into smaller pieces
3. Generating embeddings using Gemini or local embedder
4. Saving docs.jsonl and embeddings.npy to index directory

Usage:
    python tools/build_rag_index_gemini.py --input research.txt --output backend/llm_index
    python tools/build_rag_index_gemini.py --generate-sample --output backend/llm_index

Environment Variables:
    GOOGLE_API_KEY or GEMINI_API_KEY: API key for Gemini embeddings
    GEMINI_EMBED_MODEL: Embedding model name (default: models/text-embedding-004)
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from llms.backend import gemini_adapter
    import numpy as np
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Please ensure you're running from the repository root and dependencies are installed.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Sample credit risk research content for demo
SAMPLE_RESEARCH = """
Credit Risk Assessment Best Practices

1. Payment History Analysis
Regular on-time payments indicate lower credit risk. Missing even one payment can significantly increase risk.
However, a single missed payment from years ago is less concerning than recent payment issues.
The phrase "no missed payments" is a strong positive indicator, while "missed payments" signals risk.

2. Negation in Credit Analysis
It is critical to correctly interpret negation in applicant statements:
- "no late payments" = POSITIVE (low risk)
- "late payments" = NEGATIVE (high risk)
- "never defaulted" = POSITIVE
- "defaulted" = NEGATIVE
Failure to account for negation leads to incorrect risk assessment.

3. Debt-to-Income Ratio
High debt relative to income suggests difficulty meeting obligations. A ratio above 0.43 is concerning.
However, if an applicant states "low debt" or "debt paid off", this is a mitigating factor.

4. Credit Utilization
Using more than 30% of available credit indicates potential financial stress.
Statements like "rarely uses credit" or "low utilization" are positive signals.

5. Temporal Considerations
Recent financial events carry more weight than historical events:
- Recent missed payment: high risk
- Missed payment 5 years ago with clean history since: lower risk
- Recent bankruptcy: very high risk
- Bankruptcy discharged 10 years ago: moderate impact

6. Employment Stability
Stable employment history reduces risk. Job changes are normal but frequent changes may indicate instability.
Phrases like "long tenure" or "stable income" are positive.

7. Intensity Modifiers
Pay attention to frequency words:
- "rarely" misses payments: better than "sometimes"
- "often" late: worse than "occasionally"
- "always" pays on time: very positive
- "never" had issues: very positive

8. Mitigating Factors
Even with risk indicators, certain factors reduce overall risk:
- High income or savings
- Stable employment
- Co-signer or collateral
- Strong payment history despite one-time issue
- External circumstances (medical emergency) that are now resolved

9. Multiple Credit Lines
Having multiple credit accounts can be positive (diverse credit mix) or negative (over-extended).
"Multiple active loans" may indicate risk if paired with high utilization.
"Successfully managing multiple accounts" is positive.

10. Red Flags
Immediate high-risk indicators:
- Recent bankruptcy or foreclosure
- Collections accounts
- Multiple recent inquiries (credit seeking behavior)
- Delinquencies
- Very high debt-to-income ratio
However, always check for negation: "no recent bankruptcy" is good.

11. Positive Indicators
Strong low-risk signals:
- Long credit history with no negative marks
- Low credit utilization
- Diverse credit types managed well
- Stable income and employment
- Significant savings or assets
- No missed payments or defaults

12. Context Matters
A single risk factor doesn't determine creditworthiness. Consider the full picture:
- Income level relative to requested amount
- Purpose of credit
- Collateral or security
- Credit history trends (improving vs declining)
- Life circumstances and external factors
"""


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text to chunk
        chunk_size: Target size of each chunk in characters
        overlap: Overlap between consecutive chunks
        
    Returns:
        List of text chunks
    """
    # Split into paragraphs first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # If adding this paragraph would exceed chunk size, save current chunk
        if current_chunk and len(current_chunk) + len(para) > chunk_size:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap from previous
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + "\n\n" + para
            else:
                current_chunk = para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
    
    # Add final chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def build_index(
    input_path: str = None,
    output_dir: str = 'backend/llm_index',
    provider: str = 'gemini',
    chunk_size: int = 500,
    generate_sample: bool = False
):
    """
    Build RAG index from input text.
    
    Args:
        input_path: Path to input text file (or None to use sample)
        output_dir: Directory to save index artifacts
        provider: Embedding provider ('gemini' or 'local')
        chunk_size: Size of text chunks
        generate_sample: If True, use built-in sample research
    """
    # Load or generate input text
    if generate_sample or input_path is None:
        logger.info("Using sample credit risk research content")
        text = SAMPLE_RESEARCH
    else:
        logger.info(f"Loading text from {input_path}")
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            logger.error(f"Failed to read input file: {e}")
            return False
    
    # Chunk the text
    logger.info(f"Chunking text with chunk_size={chunk_size}")
    chunks = chunk_text(text, chunk_size=chunk_size)
    logger.info(f"Created {len(chunks)} chunks")
    
    # Check embedding provider availability
    if provider == 'gemini':
        if not gemini_adapter.is_gemini_available():
            logger.warning("Gemini not available, falling back to local embedder")
            provider = 'local'
    
    if provider == 'local':
        if not gemini_adapter.is_local_embedder_available():
            logger.error("No embedding provider available. Install sentence-transformers or set GOOGLE_API_KEY")
            return False
    
    # Generate embeddings
    logger.info(f"Generating embeddings using {provider} provider")
    try:
        embeddings = gemini_adapter.embed_texts(chunks, provider=provider)
        logger.info(f"Generated embeddings with shape {embeddings.shape}")
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save documents as JSONL
    docs_path = os.path.join(output_dir, 'docs.jsonl')
    logger.info(f"Saving documents to {docs_path}")
    try:
        with open(docs_path, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                doc = {
                    'id': i,
                    'text': chunk,
                    'source': input_path or 'sample_research'
                }
                f.write(json.dumps(doc) + '\n')
    except Exception as e:
        logger.error(f"Failed to save documents: {e}")
        return False
    
    # Save embeddings as numpy array
    embeddings_path = os.path.join(output_dir, 'embeddings.npy')
    logger.info(f"Saving embeddings to {embeddings_path}")
    try:
        np.save(embeddings_path, embeddings)
    except Exception as e:
        logger.error(f"Failed to save embeddings: {e}")
        return False
    
    logger.info(f"✓ Successfully built RAG index in {output_dir}")
    logger.info(f"  - {len(chunks)} documents")
    logger.info(f"  - Embedding dimension: {embeddings.shape[1]}")
    logger.info(f"  - Provider: {provider}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Build RAG index for Gemini integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build index from sample research
  python tools/build_rag_index_gemini.py --generate-sample --output backend/llm_index
  
  # Build index from custom text file
  python tools/build_rag_index_gemini.py --input research.txt --output backend/llm_index
  
  # Use local embedder instead of Gemini
  python tools/build_rag_index_gemini.py --generate-sample --provider local --output backend/llm_index
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to input text file containing research content'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='backend/llm_index',
        help='Output directory for index artifacts (default: backend/llm_index)'
    )
    parser.add_argument(
        '--provider', '-p',
        type=str,
        choices=['gemini', 'local'],
        default='gemini',
        help='Embedding provider to use (default: gemini)'
    )
    parser.add_argument(
        '--chunk-size', '-c',
        type=int,
        default=500,
        help='Size of text chunks in characters (default: 500)'
    )
    parser.add_argument(
        '--generate-sample', '-g',
        action='store_true',
        help='Use built-in sample research content instead of input file'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.generate_sample and not args.input:
        parser.error("Either --input or --generate-sample must be specified")
    
    # Build the index
    success = build_index(
        input_path=args.input,
        output_dir=args.output,
        provider=args.provider,
        chunk_size=args.chunk_size,
        generate_sample=args.generate_sample
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
