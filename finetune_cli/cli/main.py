"""
CLI entry point for the fine-tuning framework.

Commands:
  train      Fine-tune a model (LoRA or QLoRA)
  evaluate   Evaluate a saved model checkpoint
  benchmark  Run before/after benchmark comparison

Usage::

  python -m finetune_cli.cli train --config config.yaml
  python -m finetune_cli.cli train --model gpt2 --dataset ./data.jsonl
  python -m finetune_cli.cli evaluate --model-path ./output --dataset ./data.jsonl
  python -m finetune_cli.cli benchmark --base gpt2 --finetuned ./output --dataset ./data.jsonl
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from ..core.types import (
    TrainingMethod,
    DatasetSource,
    EvaluationMetric,
)
from ..core.config import ConfigBuilder
from ..core.exceptions import FineTuneError
from ..utils.logging import setup_logger, LogLevel


app = typer.Typer(
    name="finetune-cli",
    help="Production-grade LLM fine-tuning CLI (LoRA / QLoRA)",
    add_completion=False,
)
console = Console()


# ============================================================================
# TRAIN
# ============================================================================


@app.command()
def train(
    # Config file (takes precedence over individual flags)
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML/JSON config file"),
    # Quick flags (used when no config file provided)
    model: str = typer.Option("gpt2", "--model", "-m", help="HuggingFace model id"),
    dataset: Optional[Path] = typer.Option(None, "--dataset", "-d", help="Local dataset path"),
    hf_dataset: Optional[str] = typer.Option(None, "--hf-dataset", help="HuggingFace dataset id"),
    output_dir: Path = typer.Option(Path("./output"), "--output", "-o", help="Output directory"),
    method: TrainingMethod = typer.Option(TrainingMethod.LORA, "--method", help="Training method"),
    # LoRA hyper-params
    lora_r: int = typer.Option(8, "--lora-r", help="LoRA rank"),
    lora_alpha: int = typer.Option(32, "--lora-alpha", help="LoRA alpha"),
    lora_dropout: float = typer.Option(0.1, "--lora-dropout", help="LoRA dropout"),
    # Training hyper-params
    epochs: int = typer.Option(3, "--epochs", "-e", help="Number of epochs"),
    batch_size: int = typer.Option(4, "--batch-size", "-b", help="Per-device batch size"),
    lr: float = typer.Option(2e-4, "--lr", help="Learning rate"),
    max_length: int = typer.Option(512, "--max-length", help="Max token length"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples", help="Limit dataset size"),
    # Misc
    quantize_4bit: bool = typer.Option(False, "--4bit", help="Load model in 4-bit (QLoRA)"),
    fp16: bool = typer.Option(False, "--fp16", help="Mixed precision FP16"),
    log_level: str = typer.Option("info", "--log-level", help="Logging verbosity"),
):
    """Fine-tune a model using LoRA or QLoRA."""
    logger = setup_logger("cli.train", level=LogLevel(log_level))

    try:
        # --- Build config ---
        if config is not None:
            from ..core.config import PipelineConfig
            pipeline_config = (
                PipelineConfig.from_yaml(config)
                if config.suffix in (".yml", ".yaml")
                else PipelineConfig.from_json(config)
            )
        else:
            # Validate dataset source
            if dataset is None and hf_dataset is None:
                console.print("[red]Error:[/red] Provide --dataset or --hf-dataset")
                raise typer.Exit(code=1)

            ds_path = str(dataset) if dataset else str(hf_dataset)
            ds_source = DatasetSource.LOCAL_FILE if dataset else DatasetSource.HUGGINGFACE_HUB

            _LORA_METHODS = {
                TrainingMethod.LORA,
                TrainingMethod.QLORA,
                TrainingMethod.INSTRUCTION_TUNING,
                TrainingMethod.DPO,
            }
            builder = (
                ConfigBuilder()
                .with_model(model, load_in_4bit=quantize_4bit)
                .with_dataset(ds_path, source=ds_source, max_samples=max_samples)
                .with_tokenization(max_length=max_length)
                .with_training(
                    method=method,
                    output_dir=str(output_dir),
                    num_epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=lr,
                    fp16=fp16,
                )
            )
            if method in _LORA_METHODS:
                builder = builder.with_lora(
                    r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
                )
            pipeline_config = builder.build()

        console.print(Panel(f"[bold green]Training[/bold green] {pipeline_config.model.name}"))

        # --- Load model ---
        from ..models.loader import load_model_and_tokenizer
        model_obj, tokenizer = load_model_and_tokenizer(pipeline_config.model.to_config())

        # --- Prepare data ---
        from ..data import prepare_dataset
        dataset_obj = prepare_dataset(
            dataset_config=pipeline_config.dataset.to_config(),
            tokenization_config=pipeline_config.tokenization.to_config(),
            tokenizer=tokenizer,
            split_for_validation=True,
        )

        # --- Train ---
        from ..trainers import TrainerFactory
        result = TrainerFactory.train(
            model=model_obj,
            tokenizer=tokenizer,
            dataset=dataset_obj,
            training_config=pipeline_config.training.to_config(),
            lora_config=pipeline_config.lora.to_config() if pipeline_config.lora else None,
            model_config=pipeline_config.model.to_config(),
        )

        console.print(Panel(
            f"[bold green]✓ Training complete[/bold green]\n"
            f"Model saved to: {result.output_dir}\n"
            f"Train loss: {result.train_loss:.4f}\n"
            f"Steps: {result.steps_completed}"
        ))

    except FineTuneError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)


# ============================================================================
# EVALUATE
# ============================================================================


@app.command()
def evaluate(
    model_path: Path = typer.Argument(..., help="Path to fine-tuned model/adapter"),
    dataset: Optional[Path] = typer.Option(None, "--dataset", "-d"),
    hf_dataset: Optional[str] = typer.Option(None, "--hf-dataset"),
    base_model: str = typer.Option("gpt2", "--base-model", help="Base model id for adapter merging"),
    metrics: str = typer.Option("rougeL,bleu", "--metrics", help="Comma-separated metric names"),
    batch_size: int = typer.Option(4, "--batch-size"),
    num_samples: int = typer.Option(100, "--num-samples"),
    max_gen_length: int = typer.Option(100, "--max-gen-length"),
):
    """Evaluate a saved model checkpoint and print metric scores."""
    from ..models.loader import load_model_and_tokenizer
    from ..core.types import ModelConfig, DeviceType, EvaluationConfig
    from ..evaluation import BenchmarkRunner

    if dataset is None and hf_dataset is None:
        console.print("[red]Error:[/red] Provide --dataset or --hf-dataset")
        raise typer.Exit(code=1)

    # Parse metrics
    metric_enums = []
    for m in metrics.split(","):
        m = m.strip()
        try:
            metric_enums.append(EvaluationMetric(m))
        except ValueError:
            console.print(f"[yellow]Warning:[/yellow] Unknown metric '{m}', skipping")

    if not metric_enums:
        console.print("[red]Error:[/red] No valid metrics specified")
        raise typer.Exit(code=1)

    console.print(Panel(f"[bold blue]Evaluating[/bold blue] {model_path}"))

    # Load fine-tuned model
    model_config = ModelConfig(name=str(model_path), device=DeviceType.AUTO)
    model_obj, tokenizer = load_model_and_tokenizer(model_config)

    # Load evaluation dataset
    from ..data import quick_load
    ds_path = str(dataset) if dataset else str(hf_dataset)
    ds_source = "local" if dataset else "huggingface"
    eval_dataset = quick_load(ds_path, tokenizer, source=ds_source, max_samples=num_samples)

    eval_config = EvaluationConfig(
        metrics=metric_enums,
        batch_size=batch_size,
        num_samples=num_samples,
        generation_max_length=max_gen_length,
        generation_do_sample=True,
    )

    runner = BenchmarkRunner(eval_config, tokenizer)
    result = runner.evaluate(model_obj, eval_dataset, label="fine-tuned")

    console.print("\n[bold]Results:[/bold]")
    for metric, score in result.scores.items():
        console.print(f"  {metric:<20} {score:.4f}")


# ============================================================================
# BENCHMARK
# ============================================================================


@app.command()
def benchmark(
    base: str = typer.Argument(..., help="Base model id (e.g. gpt2)"),
    finetuned: Path = typer.Argument(..., help="Path to fine-tuned model/adapter"),
    dataset: Optional[Path] = typer.Option(None, "--dataset", "-d"),
    hf_dataset: Optional[str] = typer.Option(None, "--hf-dataset"),
    metrics: str = typer.Option("rougeL,bleu", "--metrics"),
    batch_size: int = typer.Option(4, "--batch-size"),
    num_samples: int = typer.Option(100, "--num-samples"),
    max_gen_length: int = typer.Option(100, "--max-gen-length"),
):
    """Compare base model vs fine-tuned model on key metrics."""
    from ..models.loader import load_model_and_tokenizer
    from ..core.types import ModelConfig, DeviceType, EvaluationConfig
    from ..evaluation import BenchmarkRunner, BenchmarkReport

    if dataset is None and hf_dataset is None:
        console.print("[red]Error:[/red] Provide --dataset or --hf-dataset")
        raise typer.Exit(code=1)

    metric_enums = [EvaluationMetric(m.strip()) for m in metrics.split(",")]

    console.print(Panel(
        f"[bold cyan]Benchmark[/bold cyan]\n"
        f"Base: {base}\n"
        f"Fine-tuned: {finetuned}"
    ))

    # Load both models
    base_cfg = ModelConfig(name=base, device=DeviceType.AUTO)
    base_model, tokenizer = load_model_and_tokenizer(base_cfg)

    ft_cfg = ModelConfig(name=str(finetuned), device=DeviceType.AUTO)
    ft_model, _ = load_model_and_tokenizer(ft_cfg)

    # Load dataset
    from ..data import quick_load
    ds_path = str(dataset) if dataset else str(hf_dataset)
    ds_source = "local" if dataset else "huggingface"
    eval_dataset = quick_load(ds_path, tokenizer, source=ds_source, max_samples=num_samples)

    eval_config = EvaluationConfig(
        metrics=metric_enums,
        batch_size=batch_size,
        num_samples=num_samples,
        generation_max_length=max_gen_length,
        generation_do_sample=True,
    )

    runner = BenchmarkRunner(eval_config, tokenizer)
    report = runner.run_comparison(base_model, ft_model, eval_dataset)

    console.print("\n" + report.summary())


# ============================================================================
# UPLOAD
# ============================================================================


@app.command()
def upload(
    model_path: Path = typer.Argument(..., help="Path to fine-tuned model/adapter to upload"),
    repo_id: str = typer.Argument(..., help="HuggingFace repo id, e.g. 'username/my-model'"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="HF API token (or set HF_TOKEN env var)"),
    private: bool = typer.Option(False, "--private", help="Make repository private"),
    commit_message: str = typer.Option("Upload fine-tuned model", "--message", "-m"),
    merge_adapter: bool = typer.Option(False, "--merge-adapter", help="Merge LoRA adapter into base model before uploading"),
    base_model: Optional[str] = typer.Option(None, "--base-model", help="Base model id (required when --merge-adapter is set)"),
):
    """Upload a fine-tuned model or LoRA adapter to HuggingFace Hub."""
    import os

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        console.print("[red]Error:[/red] huggingface_hub not installed. Run: pip install huggingface-hub")
        raise typer.Exit(code=1)

    if not model_path.exists():
        console.print(f"[red]Error:[/red] Model path does not exist: {model_path}")
        raise typer.Exit(code=1)

    # Resolve token
    resolved_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not resolved_token:
        console.print("[red]Error:[/red] No HF token provided. Use --token or set HF_TOKEN env var.")
        raise typer.Exit(code=1)

    console.print(Panel(f"[bold cyan]Uploading[/bold cyan] → {repo_id}"))
    api = HfApi(token=resolved_token)

    # Optionally merge LoRA adapter
    upload_path = model_path
    if merge_adapter:
        if not base_model:
            console.print("[red]Error:[/red] --base-model is required when using --merge-adapter")
            raise typer.Exit(code=1)
        console.print(f"Merging LoRA adapter with base model '{base_model}'...")
        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import tempfile

            base = AutoModelForCausalLM.from_pretrained(base_model)
            merged = PeftModel.from_pretrained(base, str(model_path)).merge_and_unload()
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))

            merge_dir = Path(tempfile.mkdtemp()) / "merged"
            merge_dir.mkdir(parents=True)
            merged.save_pretrained(str(merge_dir))
            tokenizer.save_pretrained(str(merge_dir))
            upload_path = merge_dir
            console.print(f"[green]✓[/green] Merged model saved temporarily to {merge_dir}")
        except Exception as exc:
            console.print(f"[red]Merge failed:[/red] {exc}")
            raise typer.Exit(code=1)

    # Create repo (no-op if already exists)
    try:
        create_repo(repo_id, token=resolved_token, private=private, exist_ok=True)
        console.print(f"[green]✓[/green] Repository ready: https://huggingface.co/{repo_id}")
    except Exception as exc:
        console.print(f"[red]Failed to create repo:[/red] {exc}")
        raise typer.Exit(code=1)

    # Upload folder
    try:
        api.upload_folder(
            folder_path=str(upload_path),
            repo_id=repo_id,
            commit_message=commit_message,
        )
        console.print(Panel(
            f"[bold green]✓ Upload complete[/bold green]\n"
            f"Model live at: https://huggingface.co/{repo_id}"
        ))
    except Exception as exc:
        console.print(f"[red]Upload failed:[/red] {exc}")
        raise typer.Exit(code=1)


# ============================================================================
# MERGE
# ============================================================================


@app.command()
def merge(
    adapter_dir: Path = typer.Argument(..., help="Path to saved LoRA adapter directory"),
    output_dir: Path = typer.Argument(..., help="Directory to save the merged standalone model"),
    base_model: str = typer.Option(..., "--base-model", "-b", help="Base HuggingFace model id"),
    dtype: str = typer.Option("float32", "--dtype", help="Torch dtype: float32 | float16 | bfloat16"),
):
    """Merge a LoRA adapter into its base model and save a standalone model.

    The merged model runs without PEFT installed and can be used directly
    with transformers or uploaded to HuggingFace Hub.

    Example:
        finetune-cli merge ./outputs/gpt2_lora ./outputs/gpt2_merged --base-model gpt2
    """
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError as exc:
        console.print(f"[red]Error:[/red] Missing dependency: {exc}")
        raise typer.Exit(code=1)

    if not adapter_dir.exists():
        console.print(f"[red]Error:[/red] Adapter directory not found: {adapter_dir}")
        raise typer.Exit(code=1)

    adapter_config = adapter_dir / "adapter_config.json"
    if not adapter_config.exists():
        console.print(
            f"[red]Error:[/red] No adapter_config.json in {adapter_dir}. "
            "Is this a valid PEFT adapter directory?"
        )
        raise typer.Exit(code=1)

    try:
        _DTYPE_MAP = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype not in _DTYPE_MAP:
            console.print(f"[red]Error:[/red] Unknown dtype '{dtype}'. Choose: float32 | float16 | bfloat16")
            raise typer.Exit(code=1)

        torch_dtype = _DTYPE_MAP[dtype]

        panel_text = (
            "[bold cyan]Merging adapter[/bold cyan]\n"
            f"Base model : {base_model}\n"
            f"Adapter    : {adapter_dir}\n"
            f"Output     : {output_dir}\n"
            f"Dtype      : {dtype}"
        )
        console.print(Panel(panel_text))

        console.print("Loading base model...")
        base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch_dtype)

        console.print("Attaching LoRA adapter...")
        peft_model = PeftModel.from_pretrained(base, str(adapter_dir))

        console.print("Merging weights and unloading adapter...")
        merged = peft_model.merge_and_unload()

        console.print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))

        output_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"Saving merged model to {output_dir}...")
        merged.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        # Verify output
        expected = ["config.json", "tokenizer.json"]
        missing = [f for f in expected if not (output_dir / f).exists()]
        if missing:
            console.print(f"[yellow]Warning:[/yellow] Expected files not found: {missing}")

        msg = (
            "[bold green]\u2713 Merge complete[/bold green]\n"
            f"Standalone model saved to: {output_dir}\n"
            f'Run with: AutoModelForCausalLM.from_pretrained("{output_dir}")'
        )
        console.print(Panel(msg))

    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Merge failed:[/red] {exc}")
        raise typer.Exit(code=1)

# ============================================================================
# PRUNE
# ============================================================================


@app.command()
def prune(
    model_path: Path = typer.Argument(..., help="Path to saved model or adapter directory"),
    output_dir: Path = typer.Option(..., "--output", "-o", help="Directory to save the pruned model"),
    sparsity: float = typer.Option(0.3, "--sparsity", "-s", help="Fraction of attention heads to prune per layer (0.0–1.0)"),
    method: str = typer.Option("heads", "--method", help="Pruning target: 'heads' or 'ffn'"),
    min_heads: int = typer.Option(1, "--min-heads", help="Minimum heads to keep per layer (prevents full layer collapse)"),
):
    """Prune low-importance attention heads from a transformer model.

    Soft structured pruning: zeroes the weights of the lowest-magnitude
    attention heads in every transformer layer. No retraining required.
    Saves the pruned model to OUTPUT_DIR, ready for inference or further
    fine-tuning.

    Example:

        finetune-cli prune ./outputs/gpt2_lora --output ./outputs/gpt2_pruned --sparsity 0.3
    """
    from finetune_cli.core.types import ModelConfig, PruningConfig
    from finetune_cli.core.exceptions import FineTuneError
    from finetune_cli.trainers.structured_pruner import StructuredPruner

    if not model_path.exists():
        console.print(f"[red]Error:[/red] Model path does not exist: {model_path}")
        raise typer.Exit(code=1)

    if not (0.0 <= sparsity < 1.0):
        console.print("[red]Error:[/red] --sparsity must be in range [0.0, 1.0)")
        raise typer.Exit(code=1)

    if method not in ("heads", "ffn"):
        console.print("[red]Error:[/red] --method must be 'heads' or 'ffn'")
        raise typer.Exit(code=1)

    panel_text = (
        "[bold cyan]Pruning Configuration[/bold cyan]\n\n"
        "[yellow]Model:[/yellow] " + str(model_path) + "\n"
        "[yellow]Output:[/yellow] " + str(output_dir) + "\n"
        f"[yellow]Sparsity:[/yellow] {sparsity:.0%}\n"
        f"[yellow]Method:[/yellow] {method}\n"
        f"[yellow]Min heads per layer:[/yellow] {min_heads}"
    )
    console.print(Panel.fit(panel_text, title="✂  Structured Pruning", border_style="cyan"))

    try:
        with console.status("[bold green]Loading model..."):
            from finetune_cli.models.loader import load_model_and_tokenizer
            model_config = ModelConfig(name=str(model_path))
            model, tokenizer = load_model_and_tokenizer(model_config)
        console.print("[green]✓[/green] Model loaded")

        pruning_config = PruningConfig(
            output_dir=output_dir,
            sparsity=sparsity,
            method=method,
            min_heads_per_layer=min_heads,
        )

        with console.status("[bold green]Pruning..."):
            pruner = StructuredPruner(model, tokenizer, pruning_config)
            result = pruner.prune()

        result_text = (
            "[bold green]✓ Pruning complete[/bold green]\n\n"
            "[yellow]Output dir:[/yellow] " + str(result.output_dir) + "\n"
            f"[yellow]Params zeroed:[/yellow] {result.zeroed_param_count:,}\n"
            f"[yellow]Sparsity achieved:[/yellow] {result.sparsity_achieved:.1%}\n"
            f"[yellow]Time:[/yellow] {result.pruning_time_seconds:.1f}s\n"
            "[yellow]Layers pruned:[/yellow] " + str(len(result.heads_pruned_per_layer))
        )
        console.print(Panel.fit(result_text, title="✂  Done", border_style="green"))

    except FineTuneError as exc:
        console.print(f"\n[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(f"\n[bold red]Unexpected error:[/bold red] {exc}")
        raise typer.Exit(code=1)

# ============================================================================
# WANDA
# ============================================================================


@app.command()
def wanda(
    model_path: Path = typer.Argument(..., help="Path to saved model or adapter directory"),
    output_dir: Path = typer.Option(..., "--output", "-o", help="Directory to save the pruned model"),
    sparsity: float = typer.Option(0.5, "--sparsity", "-s", help="Fraction of weights to zero per layer (0.0–1.0)"),
    n_samples: int = typer.Option(128, "--n-samples", help="Number of calibration forward passes"),
    seq_len: int = typer.Option(128, "--seq-len", help="Sequence length for calibration inputs"),
    dataset: Optional[Path] = typer.Option(None, "--dataset", "-d", help="Calibration dataset (.jsonl/.txt). If omitted, falls back to magnitude-only scoring."),
    row_wise: bool = typer.Option(True, "--row-wise/--global", help="Apply sparsity per output row (True) or globally (False)"),
):
    """Prune a model using WANDA (Weight AND Activation) scoring.

    WANDA is zero-shot unstructured pruning: each weight is scored by
    |W| * ||activation||_2. Weights with the lowest scores are zeroed.
    No gradient pass or retraining is required.

    Recommended sparsity: 0.3–0.5. Use --dataset for best results; without
    it, falls back to magnitude-only scoring.

    Example:

        finetune-cli wanda ./outputs/gpt2_lora --output ./outputs/gpt2_wanda \\
            --sparsity 0.5 --dataset ./data/sample.jsonl
    """
    from finetune_cli.core.types import ModelConfig, WandaConfig
    from finetune_cli.core.exceptions import FineTuneError
    from finetune_cli.trainers.wanda_pruner import WandaPruner

    if not model_path.exists():
        console.print(f"[red]Error:[/red] Model path does not exist: {model_path}")
        raise typer.Exit(code=1)

    if not (0.0 <= sparsity < 1.0):
        console.print("[red]Error:[/red] --sparsity must be in range [0.0, 1.0)")
        raise typer.Exit(code=1)

    panel_text = (
        "[bold cyan]WANDA Configuration[/bold cyan]\n\n"
        "[yellow]Model:[/yellow] " + str(model_path) + "\n"
        "[yellow]Output:[/yellow] " + str(output_dir) + "\n"
        f"[yellow]Sparsity:[/yellow] {sparsity:.0%}\n"
        f"[yellow]Calibration samples:[/yellow] {n_samples}\n"
        "[yellow]Calibration dataset:[/yellow] " + (str(dataset) if dataset else "none (magnitude-only fallback)")
    )
    console.print(Panel.fit(panel_text, title="⚡  WANDA Pruning", border_style="cyan"))

    try:
        with console.status("[bold green]Loading model..."):
            from finetune_cli.models.loader import load_model_and_tokenizer
            model_config = ModelConfig(name=str(model_path))
            model, tokenizer = load_model_and_tokenizer(model_config)
        console.print("[green]✓[/green] Model loaded")

        # Build calibration input_ids if a dataset was provided
        calib_ids = None
        if dataset is not None:
            if not dataset.exists():
                console.print(f"[red]Error:[/red] Dataset not found: {dataset}")
                raise typer.Exit(code=1)
            with console.status("[bold green]Preparing calibration data..."):
                import torch
                from finetune_cli.data import quick_load, prepare_dataset
                from finetune_cli.core.types import (
                    DatasetConfig, DatasetSource, TokenizationConfig,
                )
                ds_cfg = DatasetConfig(
                    source=DatasetSource.LOCAL_FILE,
                    path=str(dataset),
                    max_samples=n_samples,
                )
                tok_cfg = TokenizationConfig(max_length=seq_len)
                ds = prepare_dataset(ds_cfg, tok_cfg, tokenizer)
                if hasattr(ds, "with_format"):
                    ds = ds.with_format("torch")
                input_ids = torch.stack([
                    row["input_ids"] for row in ds
                    if "input_ids" in row
                ][:n_samples])
                calib_ids = input_ids
            console.print(f"[green]✓[/green] Calibration data: {calib_ids.shape[0]} samples")

        wanda_config = WandaConfig(
            output_dir=output_dir,
            sparsity=sparsity,
            n_calibration_samples=n_samples,
            calibration_seq_len=seq_len,
            use_row_wise=row_wise,
        )

        with console.status("[bold green]Pruning with WANDA..."):
            pruner = WandaPruner(model, tokenizer, wanda_config)
            result = pruner.prune(calibration_input_ids=calib_ids)

        result_text = (
            "[bold green]✓ WANDA pruning complete[/bold green]\n\n"
            "[yellow]Output dir:[/yellow] " + str(result.output_dir) + "\n"
            f"[yellow]Weights zeroed:[/yellow] {result.zeroed_param_count:,} / {result.original_param_count:,}\n"
            f"[yellow]Sparsity achieved:[/yellow] {result.sparsity_achieved:.1%}\n"
            f"[yellow]Layers pruned:[/yellow] {result.layers_pruned}\n"
            f"[yellow]Time:[/yellow] {result.pruning_time_seconds:.1f}s"
        )
        console.print(Panel.fit(result_text, title="⚡  Done", border_style="green"))

    except FineTuneError as exc:
        console.print(f"\n[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(f"\n[bold red]Unexpected error:[/bold red] {exc}")
        raise typer.Exit(code=1)

# ============================================================================
# TUI  (Sprint 25)
# ============================================================================


@app.command()
def tui() -> None:
    """Launch the interactive Textual TUI for finetune-cli."""
    try:
        import textual  # noqa: F401 — check textual is present first
    except ImportError:
        from rich.console import Console
        Console().print(
            "[red]Error:[/red] Textual is not installed.\n"
            "Install it with: [bold]pip install textual>=0.52.0[/bold]"
        )
        raise typer.Exit(code=1)

    # Textual is present — import the app (real errors surface here)
    from finetune_cli.tui.app import FinetuneApp
    FinetuneApp().run()

# ============================================================================
# ENTRY POINT
# ============================================================================


def main():
    app()


if __name__ == "__main__":
    main()


# ============================================================================
# RECOMMEND
# ============================================================================


@app.command()
def recommend(
    model: str = typer.Argument(..., help="HuggingFace model id (e.g. gpt2)"),
    dataset: Optional[Path] = typer.Option(None, "--dataset", "-d", help="Local dataset path"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save YAML config to file"),
    vram_gb: Optional[float] = typer.Option(None, "--vram", help="Available VRAM in GB (auto-detect if omitted)"),
):
    """Suggest an optimal training config based on model size and available VRAM."""
    import torch
    import yaml

    console.print(f"\n[bold cyan]Inspecting model:[/bold cyan] {model}\n")

    # Detect VRAM
    if vram_gb is None:
        if torch.cuda.is_available():
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = vram_bytes / (1024 ** 3)
            console.print(f"[green]Detected VRAM:[/green] {vram_gb:.1f} GB")
        else:
            vram_gb = 0.0
            console.print("[yellow]No GPU detected — recommending CPU-safe config.[/yellow]")

    # Estimate param count from HF config
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model)
        hidden = getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 768))
        layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 12))
        param_millions = (hidden * hidden * 12 * layers) / 1_000_000
    except Exception:
        param_millions = 124

    console.print(f"[green]Estimated parameters:[/green] ~{param_millions:.0f}M\n")

    # Decision logic
    if param_millions > 7000:
        method, lora_r, batch, fp16, grad_ckpt = "qlora", 16, 1, True, True
    elif param_millions > 1000:
        if vram_gb >= 16:
            method, lora_r, batch, fp16, grad_ckpt = "lora", 16, 4, True, False
        else:
            method, lora_r, batch, fp16, grad_ckpt = "qlora", 8, 2, True, True
    elif param_millions > 300:
        if vram_gb >= 8:
            method, lora_r, batch, fp16, grad_ckpt = "lora", 8, 4, True, False
        else:
            method, lora_r, batch, fp16, grad_ckpt = "lora", 4, 2, False, False
    else:
        if vram_gb >= 4:
            method, lora_r, batch, fp16, grad_ckpt = "lora", 8, 8, False, False
        else:
            method, lora_r, batch, fp16, grad_ckpt = "full_finetuning", 0, 4, False, False

    grad_accum = max(1, 16 // batch)
    load_4bit = method == "qlora"
    dataset_path = str(dataset) if dataset else "./data/train.jsonl"

    config_dict = {
        "model": {
            "name": model,
            "device": "auto",
            "torch_dtype": "float16" if fp16 else "float32",
            "load_in_4bit": load_4bit,
        },
        "dataset": {
            "source": "local_file",
            "path": dataset_path,
            "max_samples": None,
            "shuffle": True,
        },
        "tokenization": {"max_length": 512, "truncation": True, "padding": "max_length"},
        "training": {
            "method": method,
            "output_dir": "./output",
            "num_epochs": 3,
            "batch_size": batch,
            "gradient_accumulation_steps": grad_accum,
            "learning_rate": 2e-4 if method != "full_finetuning" else 5e-5,
            "fp16": fp16,
            "gradient_checkpointing": grad_ckpt,
            "save_strategy": "epoch",
        },
    }
    if method in ("lora", "qlora", "instruction_tuning"):
        config_dict["lora"] = {
            "r": lora_r,
            "lora_alpha": lora_r * 2,
            "lora_dropout": 0.1,
            "target_modules": None,
        }

    console.print(Panel(
        f"[bold]Recommended method:[/bold] [cyan]{method}[/cyan]\n"
        f"[bold]LoRA rank:[/bold]          {lora_r if lora_r else 'N/A'}\n"
        f"[bold]Batch size:[/bold]         {batch}\n"
        f"[bold]Grad accum:[/bold]         {grad_accum}  (effective batch: {batch * grad_accum})\n"
        f"[bold]FP16:[/bold]               {fp16}\n"
        f"[bold]4-bit quant:[/bold]        {load_4bit}\n"
        f"[bold]Grad checkpointing:[/bold] {grad_ckpt}",
        title="Recommendation",
        border_style="cyan",
    ))

    yaml_str = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(yaml_str, encoding="utf-8")
        console.print(f"\n[green]✓[/green] Config saved to [bold]{output}[/bold]")
    else:
        console.print("\n[bold]Generated config:[/bold]")
        console.print(yaml_str)