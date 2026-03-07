"""
Complete example demonstrating the data pipeline.

This script shows how to:
1. Configure dataset loading
2. Load and process data
3. Split for training/validation
4. Use different loading strategies
5. Handle errors gracefully

Run: python examples/test_data_pipeline.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer

from finetune_cli.core.config import ConfigBuilder
from finetune_cli.core.types import DatasetConfig, DatasetSource, ModelConfig, TokenizationConfig
from finetune_cli.data import (
    DataPipeline,
    DatasetAnalyzer,
    detect_columns,
    prepare_dataset,
    quick_load,
)
from finetune_cli.models.loader import load_model_and_tokenizer
from finetune_cli.utils.logging import LogLevel, setup_logger


def example_1_local_jsonl():
    """Example 1: Load local JSONL file."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Loading Local JSONL File")
    print("="*70)

    # Setup logging
    logger = setup_logger("example1", level=LogLevel.INFO)

    # Create sample data file
    import json
    sample_data = [
        {"text": "Hello, this is a sample sentence for testing."},
        {"text": "Another example with different content."},
        {"text": "Machine learning is fascinating!"},
    ] * 100  # 300 samples

    data_file = Path("./data/sample.jsonl")
    data_file.parent.mkdir(exist_ok=True)

    with open(data_file, 'w') as f:
        for item in sample_data:
            f.write(json.dumps(item) + '\n')

    logger.info(f"Created sample file: {data_file}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Quick load
    dataset = quick_load(
        path=str(data_file),
        tokenizer=tokenizer,
        source="local",
        max_samples=100,
        max_length=128
    )

    logger.info(f"Loaded {len(dataset)} samples")
    logger.info(f"Sample keys: {list(dataset[0].keys())}")
    logger.info(f"Sample input_ids length: {len(dataset[0]['input_ids'])}")

    print("\n✅ Example 1 complete!")


def example_2_huggingface_dataset():
    """Example 2: Load HuggingFace dataset."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Loading HuggingFace Dataset")
    print("="*70)

    logger = setup_logger("example2", level=LogLevel.INFO)

    # Configure dataset
    dataset_config = DatasetConfig(
        source=DatasetSource.HUGGINGFACE_HUB,
        path="wikitext",
        config_name="wikitext-2-raw-v1",
        split="train",
        max_samples=500,
        shuffle=True,
        seed=42
    )

    tokenization_config = TokenizationConfig(
        max_length=256,
        truncation=True,
        padding="max_length"
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Prepare dataset
    dataset = prepare_dataset(
        dataset_config=dataset_config,
        tokenization_config=tokenization_config,
        tokenizer=tokenizer,
        split_for_validation=False
    )

    logger.info(f"Loaded {len(dataset)} samples from WikiText")

    print("\n✅ Example 2 complete!")


def example_3_full_pipeline_with_splits():
    """Example 3: Full pipeline with train/validation splits."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Full Pipeline with Splits")
    print("="*70)

    logger = setup_logger("example3", level=LogLevel.INFO)

    # Create sample CSV file
    import pandas as pd

    df = pd.DataFrame({
        'prompt': [f"Question {i}: What is AI?" for i in range(200)],
        'response': [f"Answer {i}: AI is artificial intelligence." for i in range(200)]
    })

    csv_file = Path("./data/qa_pairs.csv")
    df.to_csv(csv_file, index=False)
    logger.info(f"Created sample CSV: {csv_file}")

    # Build configuration
    config = ConfigBuilder() \
        .with_dataset(
            str(csv_file),
            source=DatasetSource.LOCAL_FILE,
            max_samples=150,
            shuffle=True
        ) \
        .with_tokenization(max_length=128) \
        .build()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Run pipeline with splits
    splits = prepare_dataset(
        dataset_config=config.dataset.to_config(),
        tokenization_config=config.tokenization.to_config(),
        tokenizer=tokenizer,
        split_for_validation=True,
        validation_ratio=0.2
    )

    logger.info(f"Train samples: {len(splits['train'])}")
    logger.info(f"Validation samples: {len(splits['validation'])}")

    # Show sample
    sample = splits['train'][0]
    logger.info(f"Sample shape: input_ids={len(sample['input_ids'])}")

    print("\n✅ Example 3 complete!")


def example_4_custom_pipeline():
    """Example 4: Custom pipeline with analysis."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Custom Pipeline with Analysis")
    print("="*70)

    logger = setup_logger("example4", level=LogLevel.INFO)

    # Create instruction dataset
    import json

    instruction_data = [
        {
            "instruction": f"Translate to French: {word}",
            "response": f"French translation of {word}",
            "system": "You are a translation assistant."
        }
        for word in ["hello", "goodbye", "thanks"] * 50
    ]

    json_file = Path("./data/instructions.json")
    with open(json_file, 'w') as f:
        json.dump(instruction_data, f)

    logger.info(f"Created instruction dataset: {json_file}")

    # Configure
    dataset_config = DatasetConfig(
        source=DatasetSource.LOCAL_FILE,
        path=str(json_file),
        text_columns=['instruction', 'response', 'system'],
        max_samples=100,
        shuffle=True
    )

    tokenization_config = TokenizationConfig(
        max_length=256,
        truncation=True,
        padding="max_length"
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Create pipeline
    pipeline = DataPipeline(dataset_config, tokenization_config, tokenizer)

    # Run pipeline
    dataset = pipeline.run(split_for_validation=False)

    # Get statistics
    stats = pipeline.get_statistics()
    logger.info(f"Statistics: {stats}")

    # Save processed dataset
    output_dir = Path("./data/processed")
    pipeline.save_processed(output_dir)
    logger.info(f"Saved processed dataset to: {output_dir}")

    print("\n✅ Example 4 complete!")


def example_5_error_handling():
    """Example 5: Demonstrate error handling."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Error Handling")
    print("="*70)

    logger = setup_logger("example5", level=LogLevel.INFO)

    from finetune_cli.core.exceptions import (
        DatasetNotFoundError,
        EmptyDatasetError,
        NoTextColumnsError,
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Test 1: File not found
    print("\n--- Test 1: File Not Found ---")
    try:
        dataset_config = DatasetConfig(
            source=DatasetSource.LOCAL_FILE,
            path="./nonexistent.json"
        )
        prepare_dataset(
            dataset_config,
            TokenizationConfig(),
            tokenizer
        )
    except DatasetNotFoundError as e:
        logger.info(f"✓ Caught expected error: {e}")

    # Test 2: Empty dataset
    print("\n--- Test 2: Empty Dataset ---")
    empty_file = Path("./data/empty.jsonl")
    empty_file.parent.mkdir(exist_ok=True)
    empty_file.write_text("")

    try:
        quick_load(str(empty_file), tokenizer)
    except EmptyDatasetError as e:
        logger.info(f"✓ Caught expected error: {e}")

    # Test 3: No text columns
    print("\n--- Test 3: No Text Columns ---")
    import pandas as pd

    df = pd.DataFrame({
        'id': [1, 2, 3],
        'label': ['a', 'b', 'c']
    })
    numeric_file = Path("./data/numeric.csv")
    df.to_csv(numeric_file, index=False)

    try:
        quick_load(str(numeric_file), tokenizer)
    except NoTextColumnsError as e:
        logger.info(f"✓ Caught expected error: {e}")

    print("\n✅ Example 5 complete! All errors handled correctly.")


def example_6_column_detection():
    """Example 6: Column detection and validation."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Column Detection")
    print("="*70)

    logger = setup_logger("example6", level=LogLevel.INFO)

    # Create dataset with various column types
    import pandas as pd

    df = pd.DataFrame({
        'id': range(100),
        'text': [f"Sample text {i}" for i in range(100)],
        'content': [f"More content {i}" for i in range(100)],
        'label': [i % 3 for i in range(100)],
        'metadata': [f"meta_{i}" for i in range(100)]
    })

    csv_file = Path("./data/mixed_columns.csv")
    df.to_csv(csv_file, index=False)

    # Load without specifying columns (auto-detect)
    from datasets import Dataset
    dataset = Dataset.from_pandas(df)

    detected = detect_columns(dataset)
    logger.info(f"Auto-detected text columns: {detected}")

    # Analyze
    stats = DatasetAnalyzer.analyze(dataset, detected)
    logger.info(f"Analysis: {stats}")

    print("\n✅ Example 6 complete!")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("DATA PIPELINE EXAMPLES")
    print("="*70)

    try:
        example_1_local_jsonl()
        example_2_huggingface_dataset()
        example_3_full_pipeline_with_splits()
        example_4_custom_pipeline()
        example_5_error_handling()
        example_6_column_detection()

        print("\n" + "="*70)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*70)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
