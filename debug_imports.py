#!/usr/bin/env python3
"""Diagnostic script to help debug import issues in deployment."""

import sys
import os
import traceback

print("=" * 70)
print("IMPORT DIAGNOSTICS")
print("=" * 70)

print(f"\nPython version: {sys.version}")
print(f"Python path: {sys.path[:3]}...")
print(f"Current directory: {os.getcwd()}")

# Check package files exist
print("\n--- Package Structure ---")
files_to_check = [
    'llms/__init__.py',
    'llms/backend/__init__.py',
    'backend/__init__.py',
    'backend/app.py',
    'llms/backend/llm_handler.py'
]

for file in files_to_check:
    exists = "✓" if os.path.exists(file) else "✗"
    print(f"{exists} {file}")

# Test imports step by step
print("\n--- Import Tests ---")

tests = [
    ("llms", "import llms"),
    ("llms.backend", "import llms.backend"),
    ("llms.backend.llm_handler", "from llms.backend import llm_handler"),
    ("process_text_word_by_word", "from llms.backend.llm_handler import process_text_word_by_word"),
    ("backend", "import backend"),
    ("backend.app", "from backend import app"),
]

for name, import_stmt in tests:
    try:
        exec(import_stmt)
        print(f"✓ {name:30s} - {import_stmt}")
    except Exception as e:
        print(f"✗ {name:30s} - {import_stmt}")
        print(f"  Error: {e}")
        traceback.print_exc()

# Test backend.app specifically
print("\n--- backend.app Details ---")
try:
    from backend import app
    print(f"✓ Module imported successfully")
    print(f"  File: {app.__file__}")
    print(f"  Functions:")
    for attr in dir(app):
        if not attr.startswith('_') and callable(getattr(app, attr)):
            print(f"    - {attr}")
except Exception as e:
    print(f"✗ Failed to import backend.app")
    print(f"  Error: {e}")
    traceback.print_exc()

print("\n" + "=" * 70)
print("DIAGNOSTICS COMPLETE")
print("=" * 70)
