"""
Training commands for CLI.

Provides commands for training models with different methods.
"""

import typer
from typing import Optional, List
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table

from ...core.types import TrainingMethod, DatasetSource, DeviceType
from ...core.config import ConfigBuilder
from ...models.loader import load_model_and_tokenizer
from ...data import prepare_dataset
from ...trainers import train_model, get_available_methods
from ...utils.logging import setup_logger, LogLevel


app = typer.Typer()
console = Console()


# ============================================================================
# TRAIN COMMAND
# ============================================================================


@app.command()
def run(
    # Model configuration
    model_name: str = typer.Option(..., "--model", "-m", help="Model name or path"),
    
    # Dataset configuration
    dataset_path: str = typer.Option(..., "--dataset", "-d", help="Dataset path or HF identifier"),
    dataset_source: str = typer.Option("local", "--source", help="Dataset source: local or huggingface"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples", help="Limit number of samples"),
    
    # Training method
    method: str = typer.Option("lora", "--method", help="Training method: lora, qlora, full_finetuning"),
    
    # Output
    output_dir: Path = typer.Option("./outputs", "--output", "-o", help="Output directory"),
    
    # Training parameters
    epochs: int = typer.Option(3, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(4, "--batch-size", "-b", help="Batch size"),
    learning_rate: float = typer.Option(2e-4, "--lr", help="Learning rate"),
    
    # LoRA parameters (if applicable)
    lora_r: int = typer.Option(8, "--lora-r", help="LoRA rank"),
    lora_alpha: int = typer.Option(32, "--lora-alpha", help="LoRA alpha"),
    lora_dropout: float = typer.Option(0.1, "--lora-dropout", help="LoRA dropout"),
    
    # Quantization (for QLoRA)
    load_in_4bit: bool = typer.Option(False, "--4bit", help="Load model in 4-bit"),
    load_in_8bit: bool = typer.Option(False, "--8bit", help="Load model in 8-bit"),
    
    # Additional options
    validation_split: float = typer.Option(0.1, "--val-split", help="Validation split ratio"),
    max_length: int = typer.Option(512, "--max-length", help="Maximum sequence length"),
    gradient_checkpointing: bool = typer.Option(False, "--grad-checkpoint", help="Enable gradient checkpointing"),
    
    # Logging
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
    save_config: bool = typer.Option(True, "--save-config/--no-save-config", help="Save configuration"),
):
    """
    Train a model with specified configuration.
    
    Example:
        finetune-cli train --model gpt2 --dataset ./data.jsonl --method lora --epochs 3
    """
    
    # Setup logging
    logger = setup_logger("train", level=LogLevel[log_level.upper()])
    
    # Display training info
    console.print(Panel.fit(
        f"[bold cyan]Training Configuration[/bold cyan]\n\n"
        f"[yellow]Model:[/yellow] {model_name}\n"
        f"[yellow]Dataset:[/yellow] {dataset_path}\n"
        f"[yellow]Method:[/yellow] {method}\n"
        f"[yellow]Epochs:[/yellow] {epochs}\n"
        f"[yellow]Batch Size:[/yellow] {batch_size}\n"
        f"[yellow]Output:[/yellow] {output_dir}",
        title="🚀 Training",
        border_style="cyan"
    ))
    
    try:
        # Validate method
        method_enum = TrainingMethod(method)
        available = [m.value for m in get_available_methods()]
        if method not in available:
            console.print(f"[red]Error:[/red] Method '{method}' not available")
            console.print(f"Available methods: {', '.join(available)}")
            raise typer.Exit(1)
        
        # Build configuration
        with console.status("[bold green]Building configuration..."):
            config_builder = ConfigBuilder()
            
            # Model config
            config_builder.with_model(
                model_name,
                device=DeviceType.AUTO,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit
            )
            
            # Dataset config
            source = DatasetSource.LOCAL_FILE if dataset_source == "local" else DatasetSource.HUGGINGFACE_HUB
            config_builder.with_dataset(
                dataset_path,
                source=source,
                max_samples=max_samples
            )
            
            # Tokenization config
            config_builder.with_tokenization(max_length=max_length)
            
            # Training config
            config_builder.with_training(
                method_enum,
                str(output_dir),
                num_epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                gradient_checkpointing=gradient_checkpointing
            )
            
            # LoRA config (if needed)
            if method in ['lora', 'qlora']:
                config_builder.with_lora(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout
                )
            
            config = config_builder.build()
            
            # Save config if requested
            if save_config:
                config_file = output_dir / "training_config.json"
                output_dir.mkdir(parents=True, exist_ok=True)
                config.to_json(config_file)
                console.print(f"[green]✓[/green] Configuration saved to {config_file}")
        
        # Load model
        with console.status("[bold green]Loading model..."):
            model, tokenizer = load_model_and_tokenizer(config.model.to_config())
            console.print("[green]✓[/green] Model loaded successfully")
        
        # Prepare dataset
        with console.status("[bold green]Preparing dataset..."):
            splits = prepare_dataset(
                config.dataset.to_config(),
                config.tokenization.to_config(),
                tokenizer,
                split_for_validation=True,
                validation_ratio=validation_split
            )
            console.print(f"[green]✓[/green] Dataset prepared: {len(splits['train'])} train, {len(splits['validation'])} val")
        
        # Train
        console.print("\n[bold yellow]Starting training...[/bold yellow]\n")
        
        result = train_model(
            model=model,
            tokenizer=tokenizer,
            train_dataset=splits['train'],
            eval_dataset=splits['validation'],
            training_config=config.training.to_config(),
            lora_config=config.lora.to_config() if method in ['lora', 'qlora'] else None,
            model_config=config.model.to_config() if method == 'qlora' else None
        )
        
        # Display results
        console.print("\n")
        results_table = Table(title="📊 Training Results", show_header=True)
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")
        
        results_table.add_row("Final Loss", f"{result.final_loss:.4f}")
        results_table.add_row("Best Loss", f"{result.best_loss:.4f}")
        results_table.add_row("Epochs", str(result.num_epochs_completed))
        results_table.add_row("Total Steps", str(result.total_steps))
        results_table.add_row("Training Time", f"{result.training_time_seconds:.2f}s")
        results_table.add_row("Output Dir", str(result.output_dir))
        
        console.print(results_table)
        console.print("\n[bold green]✓ Training complete![/bold green]")
        
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        logger.error(f"Training failed: {e}", exc_info=True)
        raise typer.Exit(1)


# ============================================================================
# QUICK TRAIN COMMAND
# ============================================================================


@app.command()
def quick(
    model: str = typer.Argument(..., help="Model name"),
    dataset: str = typer.Argument(..., help="Dataset path"),
    output: Path = typer.Argument("./outputs", help="Output directory"),
):
    """
    Quick training with sensible defaults.
    
    Example:
        finetune-cli train quick gpt2 ./data.jsonl ./outputs
    """
    
    console.print("[bold cyan]Quick Training Mode[/bold cyan]")
    console.print("Using defaults: LoRA, 3 epochs, batch_size=4\n")
    
    # Call main train with defaults
    ctx = typer.Context(run)
    ctx.invoke(
        run,
        model_name=model,
        dataset_path=dataset,
        output_dir=output,
        method="lora",
        epochs=3,
        batch_size=4
    )


# ============================================================================
# RESUME COMMAND
# ============================================================================


@app.command()
def resume(
    checkpoint: Path = typer.Argument(..., help="Checkpoint directory"),
    additional_epochs: int = typer.Option(1, "--epochs", "-e", help="Additional epochs to train"),
):
    """
    Resume training from checkpoint.
    
    Example:
        finetune-cli train resume ./outputs/checkpoint-100 --epochs 2
    """
    
    console.print(Panel.fit(
        f"[bold cyan]Resuming Training[/bold cyan]\n\n"
        f"[yellow]Checkpoint:[/yellow] {checkpoint}\n"
        f"[yellow]Additional Epochs:[/yellow] {additional_epochs}",
        title="🔄 Resume",
        border_style="cyan"
    ))
    
    try:
        # Load config from checkpoint
        config_file = checkpoint / "training_config.json"
        if not config_file.exists():
            console.print("[red]Error:[/red] No config found in checkpoint directory")
            raise typer.Exit(1)
        
        from ...core.config import PipelineConfig
        config = PipelineConfig.from_json(config_file)
        
        # Update epochs
        from dataclasses import replace
        config.training = replace(config.training, num_epochs=additional_epochs)
        
        console.print("[green]✓[/green] Configuration loaded")
        
        # Continue training (implementation would call trainer with resume_from_checkpoint)
        console.print("[yellow]Resume functionality to be implemented in trainer[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)