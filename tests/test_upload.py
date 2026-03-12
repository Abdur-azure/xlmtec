"""
Unit tests for the `upload` CLI subcommand.

All HuggingFace Hub calls and PEFT merge operations are mocked.
No network, no GPU, no real model files required.
"""

import os
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from xlmtec.cli.main import app

runner = CliRunner()


# ============================================================================
# HELPERS
# ============================================================================


def _mock_hf_stack():
    """Patch huggingface_hub so no network calls are made."""
    mock_api = MagicMock()
    mock_api.upload_folder.return_value = None
    mock_hf_module = MagicMock()
    mock_hf_module.HfApi.return_value = mock_api
    mock_hf_module.create_repo.return_value = None
    return patch.dict("sys.modules", {"huggingface_hub": mock_hf_module}), mock_api


# ============================================================================
# TESTS
# ============================================================================


class TestUploadCommand:
    def test_missing_model_path_exits_one(self, tmp_path):
        """upload exits 1 when model_path does not exist."""
        hf_patch, _ = _mock_hf_stack()
        with hf_patch:
            result = runner.invoke(
                app,
                [
                    "upload",
                    str(tmp_path / "nonexistent"),
                    "user/repo",
                    "--token",
                    "hf_fake",
                ],
            )
        assert result.exit_code == 1

    def test_missing_token_exits_one(self, tmp_path):
        """upload exits 1 when no token is provided and HF_TOKEN is not set."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        hf_patch, _ = _mock_hf_stack()
        env = {
            k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
        }

        with hf_patch:
            with patch.dict(os.environ, env, clear=True):
                result = runner.invoke(
                    app,
                    [
                        "upload",
                        str(model_dir),
                        "user/repo",
                    ],
                )
        assert result.exit_code == 1

    def test_adapter_upload_exits_zero(self, tmp_path):
        """Standard adapter upload (no merge) exits 0."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        hf_patch, mock_api = _mock_hf_stack()
        with hf_patch:
            result = runner.invoke(
                app,
                [
                    "upload",
                    str(model_dir),
                    "user/my-adapter",
                    "--token",
                    "hf_fake",
                ],
            )

        assert result.exit_code == 0, result.output
        mock_api.upload_folder.assert_called_once()

    def test_private_flag_passed_to_create_repo(self, tmp_path):
        """--private flag is forwarded to create_repo."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        mock_api = MagicMock()
        mock_hf_module = MagicMock()
        mock_hf_module.HfApi.return_value = mock_api
        mock_create_repo = MagicMock()
        mock_hf_module.create_repo = mock_create_repo

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf_module}):
            result = runner.invoke(
                app,
                [
                    "upload",
                    str(model_dir),
                    "user/my-private-model",
                    "--token",
                    "hf_fake",
                    "--private",
                ],
            )

        assert result.exit_code == 0, result.output
        _, kwargs = mock_create_repo.call_args
        assert kwargs.get("private") is True

    def test_hf_token_env_var_accepted(self, tmp_path):
        """HF_TOKEN environment variable is used when --token is not passed."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        hf_patch, mock_api = _mock_hf_stack()
        with hf_patch:
            with patch.dict(os.environ, {"HF_TOKEN": "hf_from_env"}):
                result = runner.invoke(
                    app,
                    [
                        "upload",
                        str(model_dir),
                        "user/my-model",
                    ],
                )

        assert result.exit_code == 0, result.output

    def test_merge_adapter_requires_base_model(self, tmp_path):
        """--merge-adapter without --base-model exits 1."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        hf_patch, _ = _mock_hf_stack()
        with hf_patch:
            result = runner.invoke(
                app,
                [
                    "upload",
                    str(model_dir),
                    "user/my-model",
                    "--token",
                    "hf_fake",
                    "--merge-adapter",
                ],
            )

        assert result.exit_code == 1

    def test_merge_adapter_with_base_model_exits_zero(self, tmp_path):
        """--merge-adapter --base-model triggers merge and uploads the merged dir."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        hf_patch, mock_api = _mock_hf_stack()
        mock_base = MagicMock()
        mock_merged = MagicMock()
        mock_tokenizer = MagicMock()

        mock_peft = MagicMock()
        mock_peft.PeftModel.from_pretrained.return_value.merge_and_unload.return_value = mock_merged

        mock_transformers_local = MagicMock()
        mock_transformers_local.AutoModelForCausalLM.from_pretrained.return_value = mock_base
        mock_transformers_local.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        with hf_patch:
            with patch.dict(
                "sys.modules",
                {
                    "peft": mock_peft,
                    "transformers": mock_transformers_local,
                },
            ):
                result = runner.invoke(
                    app,
                    [
                        "upload",
                        str(model_dir),
                        "user/merged-model",
                        "--token",
                        "hf_fake",
                        "--merge-adapter",
                        "--base-model",
                        "gpt2",
                    ],
                )

        assert result.exit_code == 0, result.output
        mock_api.upload_folder.assert_called_once()
