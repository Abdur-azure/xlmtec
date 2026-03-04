"""
Recommendation commands for CLI.

Provides intelligent recommendations for training methods and configurations.
"""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ...trainers import MethodRecommender


app = typer.Typer()
console = Console()


# ============================================================================
# METHOD RECOMMENDATION
# ============================================================================


@app.command()
def method(
    model_size: float = typer.Option(..., "--model-size", help="Model size in billions (e.g., 0.124, 7, 13)"),
    vram: float = typer.Option(..., "--vram", help="Available VRAM in GB"),
    task: str = typer.Option("medium", "--task", help="Task complexity: simple, medium, complex"),
    multiple_adapters: bool = typer.Option(False, "--multiple-adapters", help="Need multiple task adapters"),
):
    """
    Get training method recommendation based on constraints.
    
    Example:
        finetune-cli recommend method --model-size 0.124 --vram 8
        finetune-cli recommend method --model-size 7 --vram 12 --task complex
    """
    
    # Convert to parameters
    model_size_params = model_size * 1e9
    
    console.print(Panel.fit(
        f"[bold cyan]Recommendation Parameters[/bold cyan]\n\n"
        f"[yellow]Model Size:[/yellow] {model_size}B parameters\n"
        f"[yellow]Available VRAM:[/yellow] {vram} GB\n"
        f"[yellow]Task Complexity:[/yellow] {task}\n"
        f"[yellow]Multiple Adapters:[/yellow] {multiple_adapters}",
        title="💡 Method Recommendation",
        border_style="cyan"
    ))
    
    # Get recommendation
    recommendation = MethodRecommender.recommend(
        model_size_params=model_size_params,
        available_vram_gb=vram,
        task_complexity=task,
        needs_multiple_adapters=multiple_adapters
    )
    
    # Display results
    if recommendation['recommendation']:
        console.print(f"\n[bold green]✓ Recommended Method:[/bold green] {recommendation['recommendation'].value}")
        console.print(f"[yellow]Reason:[/yellow] {recommendation['reason']}\n")
        
        # Memory estimates
        console.print("[bold]Memory Estimates:[/bold]")
        for method, mem in recommendation['memory_estimates'].items():
            console.print(f"  {method}: {mem}")
        
        # Alternatives
        if recommendation['alternatives']:
            console.print("\n[bold]Alternative Methods:[/bold]")
            for alt in recommendation['alternatives']:
                console.print(f"  • {alt['method'].value}: {alt['reason']}")
    else:
        console.print(f"\n[bold red]✗ {recommendation['reason']}[/bold red]")
        console.print("\n[bold]Suggestions:[/bold]")
        for suggestion in recommendation['suggestions']:
            console.print(f"  • {suggestion}")


# ============================================================================
# CONFIGURATION RECOMMENDATION
# ============================================================================


@app.command()
def config(
    method: str = typer.Argument(..., help="Training method: lora, qlora, or full_finetuning"),
    dataset_size: int = typer.Option(10000, "--dataset-size", help="Number of training samples"),
    model_size: float = typer.Option(0.124, "--model-size", help="Model size in billions"),
):
    """
    Get configuration recommendations for a training method.
    
    Example:
        finetune-cli recommend config lora --dataset-size 5000
    """
    
    console.print(f"\n[bold cyan]Configuration Recommendations for {method.upper()}[/bold cyan]\n")
    
    # Recommendations based on method and dataset size
    if method == "lora":
        table = Table(title="LoRA Configuration", show_header=True)
        table.add_column("Parameter", style="cyan")
        table.add_column("Recommended", style="green")
        table.add_column("Explanation", style="white")
        
        # Determine recommendations based on dataset size
        if dataset_size < 1000:
            lora_r = 4
            epochs = 10
            batch_size = 4
        elif dataset_size < 5000:
            lora_r = 8
            epochs = 5
            batch_size = 4
        elif dataset_size < 20000:
            lora_r = 8
            epochs = 3
            batch_size = 8
        else:
            lora_r = 16
            epochs = 2
            batch_size = 8
        
        table.add_row("lora_r", str(lora_r), "Rank determines adapter capacity")
        table.add_row("lora_alpha", str(lora_r * 4), "Typically 2-4x the rank")
        table.add_row("lora_dropout", "0.1", "Standard regularization")
        table.add_row("epochs", str(epochs), "Based on dataset size")
        table.add_row("batch_size", str(batch_size), "Balanced for memory/speed")
        table.add_row("learning_rate", "2e-4", "Standard for LoRA")
        
    elif method == "qlora":
        table = Table(title="QLoRA Configuration", show_header=True)
        table.add_column("Parameter", style="cyan")
        table.add_column("Recommended", style="green")
        table.add_column("Explanation", style="white")
        
        table.add_row("load_in_4bit", "True", "Memory efficiency")
        table.add_row("lora_r", "16", "Higher rank for quantized models")
        table.add_row("lora_alpha", "64", "4x the rank")
        table.add_row("lora_dropout", "0.1", "Standard regularization")
        table.add_row("gradient_checkpointing", "True", "Required for QLoRA")
        table.add_row("batch_size", "2-4", "Smaller for large models")
        table.add_row("learning_rate", "2e-4", "Standard")
        
    else:  # full_finetuning
        table = Table(title="Full Fine-tuning Configuration", show_header=True)
        table.add_column("Parameter", style="cyan")
        table.add_column("Recommended", style="green")
        table.add_column("Explanation", style="white")
        
        table.add_row("epochs", "3", "Fewer epochs needed")
        table.add_row("batch_size", "2", "Small to fit in memory")
        table.add_row("gradient_accumulation", "8", "Simulate larger batch")
        table.add_row("gradient_checkpointing", "True", "Save memory")
        table.add_row("learning_rate", "1e-4", "Conservative for full FT")
        table.add_row("warmup_ratio", "0.1", "Gradual warmup")
    
    console.print(table)
    
    # Additional tips
    console.print("\n[bold]Tips:[/bold]")
    console.print("  • Start with these defaults and adjust based on results")
    console.print("  • Monitor loss curves to detect under/overfitting")
    console.print("  • Use validation set to prevent overfitting")
    console.print("  • Increase epochs if loss still decreasing")


# ============================================================================
# HARDWARE RECOMMENDATION
# ============================================================================


@app.command()
def hardware(
    model_size: float = typer.Argument(..., help="Model size in billions"),
    method: str = typer.Option("lora", "--method", help="Training method"),
):
    """
    Get hardware requirements for training.
    
    Example:
        finetune-cli recommend hardware 7 --method qlora
    """
    
    console.print(f"\n[bold cyan]Hardware Requirements[/bold cyan]")
    console.print(f"Model: {model_size}B parameters, Method: {method}\n")
    
    # Estimate requirements
    param_memory = model_size * 4  # GB for FP32
    
    requirements = {
        "lora": {
            "vram": param_memory * 2,
            "system_ram": 16,
            "notes": "~50% memory savings vs full fine-tuning"
        },
        "qlora": {
            "vram": param_memory * 0.5,
            "system_ram": 16,
            "notes": "~75% memory savings, enables large models on consumer GPUs"
        },
        "full_finetuning": {
            "vram": param_memory * 4,
            "system_ram": 32,
            "notes": "Requires significant memory, best for small models"
        }
    }
    
    req = requirements.get(method, requirements["lora"])
    
    table = Table(show_header=True)
    table.add_column("Component", style="cyan")
    table.add_column("Requirement", style="yellow")
    
    table.add_row("Minimum VRAM", f"~{req['vram']:.1f} GB")
    table.add_row("Recommended VRAM", f"~{req['vram'] * 1.2:.1f} GB")
    table.add_row("System RAM", f"{req['system_ram']} GB+")
    table.add_row("GPU", "NVIDIA GPU with CUDA support")
    
    console.print(table)
    console.print(f"\n[yellow]Note:[/yellow] {req['notes']}")
    
    # GPU recommendations
    console.print("\n[bold]Recommended GPUs:[/bold]")
    
    if req['vram'] <= 8:
        console.print("  ✓ RTX 3060 (12GB)")
        console.print("  ✓ RTX 4060 Ti (16GB)")
    
    if req['vram'] <= 12:
        console.print("  ✓ RTX 3090 (24GB)")
        console.print("  ✓ RTX 4090 (24GB)")
    
    if req['vram'] <= 24:
        console.print("  ✓ A100 (40GB/80GB)")
        console.print("  ✓ H100 (80GB)")
    else:
        console.print("  ⚠️  Requires multi-GPU setup or cloud resources")