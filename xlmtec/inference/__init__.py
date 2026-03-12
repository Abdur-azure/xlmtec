"""
xlmtec.inference
~~~~~~~~~~~~~~~~~
Batch inference — run predictions over a dataset using a fine-tuned model.

Usage:
    from xlmtec.inference.predictor import BatchPredictor, PredictConfig
    result = BatchPredictor().predict(PredictConfig(
        model_dir=Path("output/run1"),
        data_path=Path("data/test.jsonl"),
        output_path=Path("predictions.jsonl"),
    ))
"""
