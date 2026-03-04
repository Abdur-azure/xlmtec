"""
Configuration management commands for CLI.

Provides commands for creating, validating, and managing configuration files.
"""

import typer
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
import json

from ...core.types import TrainingMethod, DatasetSource, EvaluationMetric
from ...core.config import ConfigBuilder, PipelineConfig


app = typer.Typer()
console = Console()


# ============================================================================
# GENERATE COMMAND
# ============================================================================


@app.command()
def generate(
    output: Path = typer.Option("./config.json", "--output", "-o", help="Output file path"),
    method: str = typer.Option("lora", "--method", help="Training method"),
    format: str = typer.Option("json", "--format", help="Config format: json or yaml"),
):
    """
    Generate a configuration template.
    
    Example:
        finetune-cli config generate --output my_config.json --method lora
    """
    
    console.print(Panel.fit(
        f"[bold cyan]Generating Configuration[/bold cyan]\n\n"
        f"[yellow]Method:[/yellow] {method}\n"
        f"[yellow]Output:[/yellow] {output}\n"
        f"[yellow]Format:[/yellow] {format}",
        title="⚙️ Config Generator",
        border_style="cyan"
    ))
    
    try:
        # Build template configuration
        method_enum = TrainingMethod(method)
        
        config = ConfigBuilder() \
            .with_model(
                "gpt2",
                device="auto"
            ) \
            .with_dataset(
                "./data.jsonl",
                source=DatasetSource.LOCAL_FILE,
                max_samples=1000
            ) \
            .with_tokenization(max_length=512) \
            .with_training(
                method_enum,
                "./outputs",
                num_epochs=3,
                batch_size=4,
                learning_rate=2e-4
            )
        
        # Add method-specific config
        if method in ['lora', 'qlora']:
            config.with_lora(r=8, lora_alpha=32, lora_dropout=0.1)
        
        # Add evaluation config
        config.with_evaluation(
            metrics=[
                EvaluationMetric.ROUGE_1,
                EvaluationMetric.ROUGE_2,
                EvaluationMetric.ROUGE_L
            ]
        )
        
        pipeline_config = config.build()
        
        # Save configuration
        output.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            pipeline_config.to_json(output)
        elif format == "yaml":
            pipeline_config.to_yaml(output)
        else:
            console.print(f"[red]Error:[/red] Unknown format '{format}'")
            raise typer.Exit(1)
        
        console.print(f"[green]✓[/green] Configuration generated: {output}")
        
        # Display preview
        console.print("\n[bold]Preview:[/bold]\n")
        
        with open(output, 'r') as f:
            content = f.read()
        
        syntax = Syntax(content, "json" if format == "json" else "yaml", theme="monokai")
        console.print(syntax)
        
        console.print("\n[bold cyan]Edit this file to customize your configuration[/bold cyan]")
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


# ============================================================================
# VALIDATE COMMAND
# ============================================================================


@app.command()
def validate(
    config_file: Path = typer.Argument(..., help="Configuration file to validate"),
):
    """
    Validate a configuration file.
    
    Example:
        finetune-cli config validate ./config.json
    """
    
    console.print(f"[bold cyan]Validating configuration:[/bold cyan] {config_file}\n")
    
    try:
        # Check file exists
        if not config_file.exists():
            console.print(f"[red]✗[/red] File not found: {config_file}")
            raise typer.Exit(1)
        
        # Load and validate
        if config_file.suffix == '.json':
            config = PipelineConfig.from_json(config_file)
        elif config_file.suffix in ['.yaml', '.yml']:
            config = PipelineConfig.from_yaml(config_file)
        else:
            console.print(f"[red]✗[/red] Unknown file format: {config_file.suffix}")
            raise typer.Exit(1)
        
        # Validation successful (Pydantic handles this)
        console.print("[green]✓[/green] Configuration is valid!\n")
        
        # Display summary
        console.print("[bold]Configuration Summary:[/bold]")
        console.print(f"  Model: {config.model.name}")
        console.print(f"  Dataset: {config.dataset.path}")
        console.print(f"  Method: {config.training.method.value}")
        console.print(f"  Epochs: {config.training.num_epochs}")
        console.print(f"  Batch Size: {config.training.batch_size}")
        
        if config.lora:
            console.print(f"  LoRA Rank: {config.lora.r}")
        
        if config.evaluation:
            metrics = [m.value for m in config.evaluation.metrics]
            console.print(f"  Metrics: {', '.join(metrics)}")
        
    except ValueError as e:
        console.print(f"[red]✗[/red] Validation failed: {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}")
        raise typer.Exit(1)


# ============================================================================
# SHOW COMMAND
# ============================================================================


@app.command()
def show(
    config_file: Path = typer.Argument(..., help="Configuration file to display"),
    section: Optional[str] = typer.Option(None, "--section", "-s", help="Show specific section"),
):
    """
    Display configuration file contents.
    
    Example:
        finetune-cli config show ./config.json
        finetune-cli config show ./config.json --section training
    """
    
    try:
        if not config_file.exists():
            console.print(f"[red]Error:[/red] File not found: {config_file}")
            raise typer.Exit(1)
        
        # Load config
        if config_file.suffix == '.json':
            config = PipelineConfig.from_json(config_file)
        else:
            config = PipelineConfig.from_yaml(config_file)
        
        config_dict = config.to_dict()
        
        # Filter by section if requested
        if section:
            if section not in config_dict:
                console.print(f"[red]Error:[/red] Unknown section '{section}'")
                console.print(f"Available sections: {', '.join(config_dict.keys())}")
                raise typer.Exit(1)
            
            display_dict = {section: config_dict[section]}
        else:
            display_dict = config_dict
        
        # Display
        content = json.dumps(display_dict, indent=2)
        syntax = Syntax(content, "json", theme="monokai")
        
        console.print(Panel(
            syntax,
            title=f"📄 {config_file.name}" + (f" - {section}" if section else ""),
            border_style="cyan"
        ))
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


# ============================================================================
# TEMPLATE COMMAND
# ============================================================================


@app.command()
def templates():
    """
    Show available configuration templates.
    
    Example:
        finetune-cli config templates
    """
    
    from rich.table import Table
    
    table = Table(title="📋 Configuration Templates", show_header=True)
    table.add_column("Template", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Command", style="yellow")
    
    templates = [
        (
            "LoRA Training",
            "Standard LoRA fine-tuning",
            "config generate --method lora"
        ),
        (
            "QLoRA Training",
            "Memory-efficient quantized LoRA",
            "config generate --method qlora"
        ),
        (
            "Full Fine-tuning",
            "Train all parameters",
            "config generate --method full_finetuning"
        ),
    ]
    
    for name, desc, cmd in templates:
        table.add_row(name, desc, cmd)
    
    console.print(table)
    console.print("\n[bold cyan]Generate a template:[/bold cyan]")
    console.print("  finetune-cli config generate --method <method> --output config.json")