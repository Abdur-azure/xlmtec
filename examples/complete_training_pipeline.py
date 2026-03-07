"""
Complete end-to-end training pipeline example.

Demonstrates the full workflow:
1. Configuration setup
2. Model loading
3. Data preparation
4. Training with different methods
5. Model saving and evaluation

Run: python examples/complete_training_pipeline.py
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from finetune_cli.core.config import ConfigBuilder
from finetune_cli.core.types import DatasetSource, DeviceType, TrainingMethod
from finetune_cli.data import prepare_dataset
from finetune_cli.models.loader import load_model_and_tokenizer
from finetune_cli.trainers import (
    MethodRecommender,
    get_available_methods,
    get_finetuning_comparison,
    train_model,
)
from finetune_cli.utils.logging import LogLevel, setup_logger

# ============================================================================
# EXAMPLE 1: LoRA Training
# ============================================================================


def example_1_lora_training():
    """Example 1: Train with LoRA on local dataset."""
    print("\n" + "="*70)
    print("EXAMPLE 1: LoRA Training")
    print("="*70)

    logger = setup_logger("example1", level=LogLevel.INFO)

    # Create sample dataset
    logger.info("Creating sample dataset...")
    sample_data = [
        {"text": f"This is training sample {i} for demonstrating LoRA fine-tuning."}
        for i in range(500)
    ]

    data_file = Path("./data/lora_train.jsonl")
    data_file.parent.mkdir(exist_ok=True, parents=True)

    with open(data_file, 'w') as f:
        for item in sample_data:
            f.write(json.dumps(item) + '\n')

    # Build configuration
    logger.info("Building configuration...")
    config = ConfigBuilder() \
        .with_model("gpt2", device=DeviceType.AUTO) \
        .with_dataset(
            str(data_file),
            source=DatasetSource.LOCAL_FILE,
            max_samples=200,
            shuffle=True
        ) \
        .with_tokenization(max_length=128) \
        .with_training(
            TrainingMethod.LORA,
            "./outputs/lora_model",
            num_epochs=2,
            batch_size=4,
            learning_rate=2e-4,
            logging_steps=10
        ) \
        .with_lora(r=8, lora_alpha=32, lora_dropout=0.1) \
        .build()

    # Load model
    logger.info("Loading model...")
    model, tokenizer = load_model_and_tokenizer(config.model.to_config())

    # Prepare dataset
    logger.info("Preparing dataset...")
    splits = prepare_dataset(
        config.dataset.to_config(),
        config.tokenization.to_config(),
        tokenizer,
        split_for_validation=True,
        validation_ratio=0.2
    )

    # Train
    logger.info("Starting training...")
    result = train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=splits['train'],
        training_config=config.training.to_config(),
        lora_config=config.lora.to_config(),
        eval_dataset=splits['validation']
    )

    # Print results
    logger.info("Training Results:")
    logger.info(f"  Final Loss: {result.final_loss:.4f}")
    logger.info(f"  Best Loss: {result.best_loss:.4f}")
    logger.info(f"  Training Time: {result.training_time_seconds:.2f}s")
    logger.info(f"  Model saved to: {result.output_dir}")

    print("\n✅ Example 1 complete!")
    return result


# ============================================================================
# EXAMPLE 2: Full Fine-tuning
# ============================================================================


def example_2_full_finetuning():
    """Example 2: Full fine-tuning on small model."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Full Fine-tuning")
    print("="*70)

    logger = setup_logger("example2", level=LogLevel.INFO)

    # Create CSV dataset
    logger.info("Creating CSV dataset...")
    import pandas as pd

    df = pd.DataFrame({
        'prompt': [f"Question {i}: What is {i}?" for i in range(300)],
        'response': [f"Answer: {i} is a number." for i in range(300)]
    })

    csv_file = Path("./data/full_ft_train.csv")
    df.to_csv(csv_file, index=False)

    # Build configuration
    config = ConfigBuilder() \
        .with_model("gpt2") \
        .with_dataset(
            str(csv_file),
            source=DatasetSource.LOCAL_FILE,
            max_samples=150
        ) \
        .with_tokenization(max_length=128) \
        .with_training(
            TrainingMethod.FULL_FINETUNING,
            "./outputs/full_ft_model",
            num_epochs=2,
            batch_size=2,
            gradient_checkpointing=True,
            logging_steps=10
        ) \
        .build()

    # Load and train
    logger.info("Loading model...")
    model, tokenizer = load_model_and_tokenizer(config.model.to_config())

    logger.info("Preparing dataset...")
    dataset = prepare_dataset(
        config.dataset.to_config(),
        config.tokenization.to_config(),
        tokenizer
    )

    logger.info("Starting training...")
    result = train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        training_config=config.training.to_config()
    )

    logger.info(f"Training complete! Loss: {result.final_loss:.4f}")

    print("\n✅ Example 2 complete!")
    return result


# ============================================================================
# EXAMPLE 3: Method Recommendation
# ============================================================================


def example_3_method_recommendation():
    """Example 3: Get training method recommendation."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Method Recommendation")
    print("="*70)

    logger = setup_logger("example3", level=LogLevel.INFO)

    # Show available methods
    available = get_available_methods()
    logger.info(f"Available training methods: {[m.value for m in available]}")

    # Get recommendations for different scenarios
    scenarios = [
        {
            "name": "Small model on consumer GPU",
            "model_size_params": 124e6,  # GPT-2 small
            "available_vram_gb": 8.0,
            "task_complexity": "medium"
        },
        {
            "name": "Medium model on high-end GPU",
            "model_size_params": 1.5e9,  # GPT-2 XL
            "available_vram_gb": 24.0,
            "task_complexity": "complex"
        },
        {
            "name": "Large model on consumer GPU",
            "model_size_params": 7e9,  # 7B model
            "available_vram_gb": 12.0,
            "task_complexity": "medium"
        }
    ]

    for scenario in scenarios:
        logger.info(f"\nScenario: {scenario['name']}")
        logger.info(f"  Model size: {scenario['model_size_params']/1e9:.1f}B params")
        logger.info(f"  Available VRAM: {scenario['available_vram_gb']}GB")

        recommendation = MethodRecommender.recommend(
            model_size_params=scenario['model_size_params'],
            available_vram_gb=scenario['available_vram_gb'],
            task_complexity=scenario['task_complexity']
        )

        if recommendation['recommendation']:
            logger.info(f"  Recommendation: {recommendation['recommendation'].value}")
            logger.info(f"  Reason: {recommendation['reason']}")

            if recommendation['alternatives']:
                alts = [alt['method'].value for alt in recommendation['alternatives']]
                logger.info(f"  Alternatives: {alts}")
        else:
            logger.info(f"  {recommendation['reason']}")
            logger.info(f"  Suggestions: {recommendation['suggestions']}")

    print("\n✅ Example 3 complete!")


# ============================================================================
# EXAMPLE 4: Training Comparison
# ============================================================================


def example_4_training_comparison():
    """Example 4: Compare different training methods."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Training Methods Comparison")
    print("="*70)

    logger = setup_logger("example4", level=LogLevel.INFO)

    comparison = get_finetuning_comparison()

    logger.info("\nTraining Methods Comparison:")
    for method, details in comparison.items():
        if method == 'recommendations':
            continue

        logger.info(f"\n{method.upper()}:")
        logger.info(f"  Trainable params: {details['trainable_params']}")
        logger.info(f"  Memory usage: {details['memory_usage']}")
        logger.info(f"  Training speed: {details['training_speed']}")
        logger.info(f"  Adaptation quality: {details['adaptation_quality']}")
        logger.info(f"  Best for: {', '.join(details['best_for'])}")

    logger.info("\nRecommendations:")
    for key, value in comparison['recommendations'].items():
        logger.info(f"  {key}: {value}")

    print("\n✅ Example 4 complete!")


# ============================================================================
# EXAMPLE 5: Resume Training from Checkpoint
# ============================================================================


def example_5_resume_training():
    """Example 5: Resume training from checkpoint."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Resume Training from Checkpoint")
    print("="*70)

    logger = setup_logger("example5", level=LogLevel.INFO)

    # Create dataset
    sample_data = [
        {"text": f"Sample text {i} for checkpoint test."}
        for i in range(200)
    ]

    data_file = Path("./data/checkpoint_test.jsonl")
    data_file.parent.mkdir(exist_ok=True, parents=True)

    with open(data_file, 'w') as f:
        for item in sample_data:
            f.write(json.dumps(item) + '\n')

    # Build config
    config = ConfigBuilder() \
        .with_model("gpt2") \
        .with_dataset(str(data_file), max_samples=100) \
        .with_tokenization(max_length=128) \
        .with_training(
            TrainingMethod.LORA,
            "./outputs/checkpoint_model",
            num_epochs=3,
            batch_size=4,
            save_strategy="epoch"
        ) \
        .with_lora(r=4, lora_alpha=16) \
        .build()

    # Load model and data
    model, tokenizer = load_model_and_tokenizer(config.model.to_config())
    dataset = prepare_dataset(
        config.dataset.to_config(),
        config.tokenization.to_config(),
        tokenizer
    )

    # Train (will save checkpoints each epoch)
    logger.info("Training with checkpoints...")
    result = train_model(
        model, tokenizer, dataset,
        config.training.to_config(),
        config.lora.to_config()
    )

    logger.info(f"Training complete! Checkpoints saved in {result.output_dir}")

    # Note: To resume, you would:
    # result = train_model(..., resume_from_checkpoint=Path("./outputs/checkpoint_model/checkpoint-X"))

    print("\n✅ Example 5 complete!")


# ============================================================================
# EXAMPLE 6: Complete Pipeline with Config File
# ============================================================================


def example_6_config_file():
    """Example 6: Load configuration from file."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Training with Config File")
    print("="*70)

    logger = setup_logger("example6", level=LogLevel.INFO)

    # Create config file
    config_path = Path("./configs/training_config.json")
    config_path.parent.mkdir(exist_ok=True, parents=True)

    # Build and save config
    config = ConfigBuilder() \
        .with_model("gpt2") \
        .with_dataset(
            "./data/lora_train.jsonl",
            source=DatasetSource.LOCAL_FILE,
            max_samples=100
        ) \
        .with_tokenization(max_length=128) \
        .with_training(
            TrainingMethod.LORA,
            "./outputs/config_test",
            num_epochs=1,
            batch_size=4
        ) \
        .with_lora(r=8, lora_alpha=32) \
        .build()

    config.to_json(config_path)
    logger.info(f"Saved configuration to: {config_path}")

    # Load config from file
    from finetune_cli.core.config import PipelineConfig
    loaded_config = PipelineConfig.from_json(config_path)
    logger.info("Loaded configuration from file")

    # Train using loaded config
    model, tokenizer = load_model_and_tokenizer(loaded_config.model.to_config())
    dataset = prepare_dataset(
        loaded_config.dataset.to_config(),
        loaded_config.tokenization.to_config(),
        tokenizer
    )

    result = train_model(
        model, tokenizer, dataset,
        loaded_config.training.to_config(),
        loaded_config.lora.to_config()
    )

    logger.info(f"Training complete using config file! Loss: {result.final_loss:.4f}")

    print("\n✅ Example 6 complete!")


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("COMPLETE TRAINING PIPELINE EXAMPLES")
    print("="*70)

    try:
        # Run examples
        example_1_lora_training()
        example_2_full_finetuning()
        example_3_method_recommendation()
        example_4_training_comparison()
        example_5_resume_training()
        example_6_config_file()

        print("\n" + "="*70)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nTrained models saved in:")
        print("  - ./outputs/lora_model/")
        print("  - ./outputs/full_ft_model/")
        print("  - ./outputs/checkpoint_model/")
        print("  - ./outputs/config_test/")

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
