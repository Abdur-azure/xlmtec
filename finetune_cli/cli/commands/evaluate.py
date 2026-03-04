"""
Evaluation commands for CLI.

Provides commands for evaluating and benchmarking models.
"""

import typer
from typing import Optional, List
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ...core.types import EvaluationMetric, EvaluationConfig
from ...models.loader import load_model_and_tokenizer
from ...data import prepare_dataset
from ...evaluation import evaluate_model, benchmark_models, ReportGenerator
from ...utils.logging import setup_logger, LogLevel


app = typer.Typer()
console = Console()


# ============================================================================
# EVALUATE COMMAND
# ============================================================================


@app.command()
def run(
    # Model
    model_path: Path = typer.Option(..., "--model", "-m", help="Model path or name"),
    
    # Dataset
    dataset_path: str = typer.Option(..., "--dataset", "-d", help="Test dataset path"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples", help="Limit samples"),
    
    # Metrics
    metrics: List[str] = typer.Option(
        ["rouge1", "rouge2", "rougeL"],
        "--metric",
        help="Metrics to compute (can specify multiple)"
    ),
    
    # Generation parameters
    max_length: int = typer.Option(100, "--max-length", help="Max generation length"),
    temperature: float = typer.Option(0.7, "--temperature", help="Generation temperature"),
    batch_size: int = typer.Option(8, "--batch-size", "-b", help="Batch size"),
    
    # Output
    save_report: Optional[Path] = typer.Option(None, "--report", help="Save report to file"),
    report_format: str = typer.Option("markdown", "--format", help="Report format: markdown, json, html"),
    
    # Logging
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
):
    """
    Evaluate a trained model.
    
    Example:
        finetune-cli evaluate --model ./outputs --dataset ./test.jsonl --metric rouge1 --metric bleu
    """
    
    logger = setup_logger("evaluate", level=LogLevel[log_level.upper()])
    
    # Display info
    console.print(Panel.fit(
        f"[bold cyan]Evaluation Configuration[/bold cyan]\n\n"
        f"[yellow]Model:[/yellow] {model_path}\n"
        f"[yellow]Dataset:[/yellow] {dataset_path}\n"
        f"[yellow]Metrics:[/yellow] {', '.join(metrics)}\n"
        f"[yellow]Samples:[/yellow] {max_samples or 'all'}",
        title="📊 Evaluation",
        border_style="cyan"
    ))
    
    try:
        # Build evaluation config
        metric_enums = []
        for m in metrics:
            try:
                metric_enums.append(EvaluationMetric(m))
            except ValueError:
                console.print(f"[yellow]Warning:[/yellow] Unknown metric '{m}', skipping")
        
        if not metric_enums:
            console.print("[red]Error:[/red] No valid metrics specified")
            raise typer.Exit(1)
        
        eval_config = EvaluationConfig(
            metrics=metric_enums,
            batch_size=batch_size,
            num_samples=max_samples,
            generation_max_length=max_length,
            generation_temperature=temperature
        )
        
        # Load model
        with console.status("[bold green]Loading model..."):
            from ...core.types import ModelConfig
            model_config = ModelConfig(name=str(model_path))
            model, tokenizer = load_model_and_tokenizer(model_config)
            console.print("[green]✓[/green] Model loaded")
        
        # Load dataset
        with console.status("[bold green]Loading dataset..."):
            from ...core.types import DatasetConfig, DatasetSource, TokenizationConfig
            
            dataset_config = DatasetConfig(
                source=DatasetSource.LOCAL_FILE,
                path=dataset_path,
                max_samples=max_samples
            )
            
            tokenization_config = TokenizationConfig(max_length=512)
            
            from ...data import load_dataset_from_config
            test_dataset = load_dataset_from_config(dataset_config)
            console.print(f"[green]✓[/green] Dataset loaded: {len(test_dataset)} samples")
        
        # Evaluate
        console.print("\n[bold yellow]Evaluating model...[/bold yellow]\n")
        
        result = evaluate_model(model, tokenizer, test_dataset, eval_config)
        
        # Display results
        results_table = Table(title="📊 Evaluation Results", show_header=True)
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Score", style="green")
        
        for metric_name, score in result.metrics.items():
            results_table.add_row(metric_name, f"{score:.4f}")
        
        results_table.add_row("", "", end_section=True)
        results_table.add_row("Samples Evaluated", str(result.num_samples))
        results_table.add_row("Evaluation Time", f"{result.evaluation_time_seconds:.2f}s")
        
        console.print(results_table)
        
        # Save report if requested
        if save_report:
            console.print(f"\n[bold cyan]Saving report...[/bold cyan]")
            
            # Create comparison result for report
            from ...evaluation.base import ComparisonResult
            from datetime import datetime
            
            comparison = ComparisonResult(
                base_metrics={},
                finetuned_metrics=result.metrics,
                improvements={},
                timestamp=datetime.now()
            )
            
            ReportGenerator.save_report(
                comparison,
                save_report,
                format=report_format,
                title="Model Evaluation Report"
            )
            console.print(f"[green]✓[/green] Report saved to {save_report}")
        
        console.print("\n[bold green]✓ Evaluation complete![/bold green]")
        
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise typer.Exit(1)


# ============================================================================
# BENCHMARK COMMAND
# ============================================================================


@app.command()
def benchmark(
    # Models
    base_model: str = typer.Option(..., "--base", help="Base model name or path"),
    finetuned_model: Path = typer.Option(..., "--finetuned", "-f", help="Fine-tuned model path"),
    
    # Dataset
    dataset_path: str = typer.Option(..., "--dataset", "-d", help="Test dataset path"),
    max_samples: Optional[int] = typer.Option(100, "--max-samples", help="Limit samples"),
    
    # Metrics
    metrics: List[str] = typer.Option(
        ["rouge1", "rouge2", "rougeL"],
        "--metric",
        help="Metrics to compute"
    ),
    
    # Output
    report_path: Path = typer.Option("./benchmark_report.md", "--report", "-r", help="Report output path"),
    report_format: str = typer.Option("markdown", "--format", help="Report format"),
    
    # Logging
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
):
    """
    Benchmark base vs fine-tuned model.
    
    Example:
        finetune-cli evaluate benchmark --base gpt2 --finetuned ./outputs --dataset ./test.jsonl
    """
    
    logger = setup_logger("benchmark", level=LogLevel[log_level.upper()])
    
    console.print(Panel.fit(
        f"[bold cyan]Benchmarking Configuration[/bold cyan]\n\n"
        f"[yellow]Base Model:[/yellow] {base_model}\n"
        f"[yellow]Fine-tuned:[/yellow] {finetuned_model}\n"
        f"[yellow]Dataset:[/yellow] {dataset_path}\n"
        f"[yellow]Metrics:[/yellow] {', '.join(metrics)}",
        title="🔬 Benchmark",
        border_style="cyan"
    ))
    
    try:
        # Build config
        metric_enums = [EvaluationMetric(m) for m in metrics]
        eval_config = EvaluationConfig(
            metrics=metric_enums,
            batch_size=8,
            num_samples=max_samples
        )
        
        # Load models
        with console.status("[bold green]Loading base model..."):
            from ...core.types import ModelConfig
            base_config = ModelConfig(name=base_model)
            base_model_obj, tokenizer = load_model_and_tokenizer(base_config)
            console.print("[green]✓[/green] Base model loaded")
        
        with console.status("[bold green]Loading fine-tuned model..."):
            ft_config = ModelConfig(name=str(finetuned_model))
            ft_model_obj, _ = load_model_and_tokenizer(ft_config)
            console.print("[green]✓[/green] Fine-tuned model loaded")
        
        # Load dataset
        with console.status("[bold green]Loading dataset..."):
            from ...core.types import DatasetConfig, DatasetSource
            dataset_config = DatasetConfig(
                source=DatasetSource.LOCAL_FILE,
                path=dataset_path,
                max_samples=max_samples
            )
            from ...data import load_dataset_from_config
            test_dataset = load_dataset_from_config(dataset_config)
            console.print(f"[green]✓[/green] Dataset loaded: {len(test_dataset)} samples")
        
        # Benchmark
        console.print("\n[bold yellow]Benchmarking models...[/bold yellow]\n")
        
        result = benchmark_models(
            base_model=base_model_obj,
            finetuned_model=ft_model_obj,
            tokenizer=tokenizer,
            dataset=test_dataset,
            config=eval_config,
            save_report=report_path
        )
        
        # Display results
        console.print("\n")
        
        comparison_table = Table(title="🔬 Benchmark Results", show_header=True)
        comparison_table.add_column("Metric", style="cyan")
        comparison_table.add_column("Base Model", style="yellow")
        comparison_table.add_column("Fine-tuned", style="green")
        comparison_table.add_column("Improvement", style="magenta")
        
        for metric in result.base_metrics:
            base_score = result.base_metrics[metric]
            ft_score = result.finetuned_metrics[metric]
            improvement = result.improvements[metric]
            
            comparison_table.add_row(
                metric,
                f"{base_score:.4f}",
                f"{ft_score:.4f}",
                f"{improvement:+.2f}%"
            )
        
        comparison_table.add_row("", "", "", "", end_section=True)
        comparison_table.add_row(
            "[bold]Average",
            "",
            "",
            f"[bold]{result.get_average_improvement():+.2f}%"
        )
        
        console.print(comparison_table)
        console.print(f"\n[green]✓[/green] Report saved to {report_path}")
        console.print("\n[bold green]✓ Benchmark complete![/bold green]")
        
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        raise typer.Exit(1)


# ============================================================================
# QUICK EVAL COMMAND
# ============================================================================


@app.command()
def quick(
    model: Path = typer.Argument(..., help="Model path"),
    dataset: str = typer.Argument(..., help="Dataset path"),
):
    """
    Quick evaluation with default metrics.
    
    Example:
        finetune-cli evaluate quick ./outputs ./test.jsonl
    """
    
    console.print("[bold cyan]Quick Evaluation Mode[/bold cyan]")
    console.print("Using default metrics: ROUGE-1, ROUGE-2, ROUGE-L\n")
    
    ctx = typer.Context(run)
    ctx.invoke(
        run,
        model_path=model,
        dataset_path=dataset,
        metrics=["rouge1", "rouge2", "rougeL"],
        max_samples=100
    )