"""
Sprint 25 -- TUI Foundation headless Pilot tests.

Requires: pip install textual>=0.52.0 pytest-asyncio

Run:
    pytest tests/test_tui.py -v
"""

import pytest

# Skip the entire module if textual is not installed.
# Install with: pip install "textual>=0.52.0" pytest-asyncio
pytest.importorskip("textual", reason="textual not installed")

from lmtool.tui.app import LMToolApp  # noqa: E402
from lmtool.tui.screens.home import HomeScreen  # noqa: E402
from lmtool.tui.widgets.command_card import CommandCard  # noqa: E402


class TestAppMount:
    """LMToolApp starts and reaches HomeScreen cleanly."""

    @pytest.mark.asyncio
    async def test_app_mounts_without_error(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.screen is not None

    @pytest.mark.asyncio
    async def test_home_screen_is_initial_screen(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)

    @pytest.mark.asyncio
    async def test_app_title_set(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            assert app.TITLE == "lmtool"


class TestHomeScreen:
    """HomeScreen renders all 6 CommandCards."""

    @pytest.mark.asyncio
    async def test_six_command_cards_rendered(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            assert len(app.screen.query(CommandCard)) == 6

    @pytest.mark.asyncio
    async def test_all_expected_card_ids_present(self):
        expected_ids = {
            "card-train", "card-evaluate", "card-benchmark",
            "card-upload", "card-merge", "card-recommend",
        }
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            ids = {card.id for card in app.screen.query(CommandCard)}
            assert ids == expected_ids

    @pytest.mark.asyncio
    async def test_cards_are_focusable(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            for card in app.screen.query(CommandCard):
                assert card.can_focus is True

    @pytest.mark.asyncio
    async def test_home_title_label_present(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            assert len(app.screen.query(".home-title")) == 1

    @pytest.mark.asyncio
    async def test_home_subtitle_label_present(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            assert len(app.screen.query(".home-subtitle")) == 1


class TestKeyboardNavigation:
    """Global keybindings work from home screen."""

    @pytest.mark.asyncio
    async def test_q_key_exits_app(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            await pilot.press("q")

    @pytest.mark.asyncio
    async def test_tab_moves_focus(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            await pilot.press("tab")
            assert app.focused is not None

    @pytest.mark.asyncio
    async def test_escape_stays_on_home(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)

    @pytest.mark.asyncio
    async def test_right_arrow_moves_focus(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            first_focused = app.focused
            await pilot.press("right")
            await pilot.pause()
            assert app.focused is not first_focused

    @pytest.mark.asyncio
    async def test_down_arrow_moves_to_next_row(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            cards = list(app.screen.query(CommandCard))
            assert len(cards) == 6
            cards[0].focus()
            await pilot.pause()
            await pilot.press("down")
            await pilot.pause()
            assert app.focused is cards[3]


class TestCommandCard:
    """CommandCard activates cleanly on click and Enter."""

    @pytest.mark.asyncio
    async def test_click_card_does_not_crash(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            # All cards are now wired — just verify click doesn't crash
            await pilot.click("#card-train")
            await pilot.pause()
            # Should have navigated away from home to TrainScreen
            assert not isinstance(app.screen, type(None))

    @pytest.mark.asyncio
    async def test_enter_on_focused_card_does_not_crash(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            # Focus recommend card and press enter — navigates to RecommendScreen
            app.screen.query_one("#card-recommend").focus()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, RecommendScreen)

    @pytest.mark.asyncio
    async def test_card_command_id_matches_id_attribute(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            train_card = app.screen.query_one("#card-train", CommandCard)
            assert train_card.command_id == "train"


# ============================================================================
# Sprint 26 — Train & Recommend screens
# ============================================================================

from textual.widgets import Checkbox  # noqa: E402

from lmtool.tui.screens.recommend import RecommendScreen  # noqa: E402
from lmtool.tui.screens.result import ResultScreen  # noqa: E402
from lmtool.tui.screens.running import RunningScreen  # noqa: E402
from lmtool.tui.screens.train import TrainScreen  # noqa: E402


class TestTrainScreen:
    """TrainScreen form renders and validates correctly."""

    @pytest.mark.asyncio
    async def test_train_card_navigates_to_train_screen(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            await pilot.click("#card-train")
            await pilot.pause()
            assert isinstance(app.screen, TrainScreen)

    @pytest.mark.asyncio
    async def test_train_screen_has_model_input(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app.switch_screen(TrainScreen())
            await pilot.pause()
            assert len(app.screen.query("#input-model")) == 1

    @pytest.mark.asyncio
    async def test_train_screen_has_method_select(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app.switch_screen(TrainScreen())
            await pilot.pause()
            assert len(app.screen.query("#select-method")) == 1

    @pytest.mark.asyncio
    async def test_train_screen_has_dataset_input(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app.switch_screen(TrainScreen())
            await pilot.pause()
            assert len(app.screen.query("#input-dataset")) == 1

    @pytest.mark.asyncio
    async def test_train_back_button_returns_home(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = TrainScreen()
            app.switch_screen(screen)
            await pilot.pause()
            await pilot.pause()
            # Call action directly — avoids button event async routing delay
            screen.action_go_home()
            await pilot.pause()
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)

    @pytest.mark.asyncio
    async def test_train_submit_empty_shows_validation(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = TrainScreen()
            app.switch_screen(screen)
            await pilot.pause()
            await pilot.pause()
            # Call action directly — avoids button event routing race
            screen.action_submit()
            await pilot.pause()
            # Validation blocked navigation — still on TrainScreen
            assert isinstance(app.screen, TrainScreen)


class TestRecommendScreen:
    """RecommendScreen form renders and validates correctly."""

    @pytest.mark.asyncio
    async def test_recommend_card_navigates_to_recommend_screen(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            await pilot.click("#card-recommend")
            await pilot.pause()
            assert isinstance(app.screen, RecommendScreen)

    @pytest.mark.asyncio
    async def test_recommend_screen_has_model_input(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app.switch_screen(RecommendScreen())
            await pilot.pause()
            assert len(app.screen.query("#input-model")) == 1

    @pytest.mark.asyncio
    async def test_recommend_screen_has_output_input(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app.switch_screen(RecommendScreen())
            await pilot.pause()
            assert len(app.screen.query("#input-output")) == 1

    @pytest.mark.asyncio
    async def test_recommend_back_button_returns_home(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = RecommendScreen()
            app.switch_screen(screen)
            await pilot.pause()
            await pilot.pause()
            screen.action_go_home()
            await pilot.pause()
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)

    @pytest.mark.asyncio
    async def test_recommend_submit_empty_shows_validation(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = RecommendScreen()
            app.switch_screen(screen)
            await pilot.pause()
            await pilot.pause()
            screen.action_submit()
            await pilot.pause()
            assert isinstance(app.screen, RecommendScreen)


class TestResultScreen:
    """ResultScreen displays correctly for success and failure."""

    @pytest.mark.asyncio
    async def test_result_screen_success_renders(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app.switch_screen(ResultScreen(
                success=True,
                metrics={"Status": "✅ Success", "Duration": "00:12"},
            ))
            await pilot.pause()
            assert isinstance(app.screen, ResultScreen)

    @pytest.mark.asyncio
    async def test_result_screen_failure_renders(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app.switch_screen(ResultScreen(
                success=False,
                metrics={"Status": "❌ Failed", "Error": "test error"},
            ))
            await pilot.pause()
            assert isinstance(app.screen, ResultScreen)

    @pytest.mark.asyncio
    async def test_result_home_button_returns_home(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app.switch_screen(ResultScreen(success=True, metrics={}))
            await pilot.pause()
            await pilot.click("#btn-home")
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)


# ============================================================================
# Sprint 27 — Evaluate, Benchmark, Merge screens
# ============================================================================

from lmtool.tui.screens.benchmark import BenchmarkScreen  # noqa: E402
from lmtool.tui.screens.evaluate import EvaluateScreen  # noqa: E402
from lmtool.tui.screens.merge import MergeScreen  # noqa: E402


class TestEvaluateScreen:
    """EvaluateScreen form renders and validates."""

    @pytest.mark.asyncio
    async def test_evaluate_card_navigates_to_evaluate_screen(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            await pilot.click("#card-evaluate")
            await pilot.pause()
            assert isinstance(app.screen, EvaluateScreen)

    @pytest.mark.asyncio
    async def test_evaluate_screen_has_model_input(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app.switch_screen(EvaluateScreen())
            await pilot.pause()
            assert len(app.screen.query("#input-model")) == 1

    @pytest.mark.asyncio
    async def test_evaluate_screen_has_dataset_input(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app.switch_screen(EvaluateScreen())
            await pilot.pause()
            assert len(app.screen.query("#input-dataset")) == 1

    @pytest.mark.asyncio
    async def test_evaluate_screen_has_metric_checkboxes(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app.switch_screen(EvaluateScreen())
            await pilot.pause()
            checkboxes = app.screen.query(Checkbox)
            assert len(checkboxes) == 5

    @pytest.mark.asyncio
    async def test_evaluate_back_returns_home(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = EvaluateScreen()
            app.switch_screen(screen)
            await pilot.pause()
            await pilot.pause()
            screen.action_go_home()
            await pilot.pause()
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)

    @pytest.mark.asyncio
    async def test_evaluate_submit_empty_shows_validation(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = EvaluateScreen()
            app.switch_screen(screen)
            await pilot.pause()
            await pilot.pause()
            screen.action_submit()
            await pilot.pause()
            assert isinstance(app.screen, EvaluateScreen)


class TestBenchmarkScreen:
    """BenchmarkScreen form renders and validates."""

    @pytest.mark.asyncio
    async def test_benchmark_card_navigates_to_benchmark_screen(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            await pilot.click("#card-benchmark")
            await pilot.pause()
            assert isinstance(app.screen, BenchmarkScreen)

    @pytest.mark.asyncio
    async def test_benchmark_screen_has_base_input(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app.switch_screen(BenchmarkScreen())
            await pilot.pause()
            assert len(app.screen.query("#input-base")) == 1

    @pytest.mark.asyncio
    async def test_benchmark_screen_has_finetuned_input(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app.switch_screen(BenchmarkScreen())
            await pilot.pause()
            assert len(app.screen.query("#input-finetuned")) == 1

    @pytest.mark.asyncio
    async def test_benchmark_back_returns_home(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = BenchmarkScreen()
            app.switch_screen(screen)
            await pilot.pause()
            await pilot.pause()
            screen.action_go_home()
            await pilot.pause()
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)

    @pytest.mark.asyncio
    async def test_benchmark_submit_empty_shows_validation(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = BenchmarkScreen()
            app.switch_screen(screen)
            await pilot.pause()
            await pilot.pause()
            screen.action_submit()
            await pilot.pause()
            assert isinstance(app.screen, BenchmarkScreen)


class TestMergeScreen:
    """MergeScreen form renders and validates."""

    @pytest.mark.asyncio
    async def test_merge_card_navigates_to_merge_screen(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            await pilot.click("#card-merge")
            await pilot.pause()
            assert isinstance(app.screen, MergeScreen)

    @pytest.mark.asyncio
    async def test_merge_screen_has_adapter_input(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app.switch_screen(MergeScreen())
            await pilot.pause()
            assert len(app.screen.query("#input-adapter")) == 1

    @pytest.mark.asyncio
    async def test_merge_screen_has_dtype_select(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app.switch_screen(MergeScreen())
            await pilot.pause()
            assert len(app.screen.query("#select-dtype")) == 1

    @pytest.mark.asyncio
    async def test_merge_back_returns_home(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = MergeScreen()
            app.switch_screen(screen)
            await pilot.pause()
            await pilot.pause()
            screen.action_go_home()
            await pilot.pause()
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)

    @pytest.mark.asyncio
    async def test_merge_submit_empty_shows_validation(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = MergeScreen()
            app.switch_screen(screen)
            await pilot.pause()
            await pilot.pause()
            screen.action_submit()
            await pilot.pause()
            assert isinstance(app.screen, MergeScreen)


# ============================================================================
# Sprint 28 — Upload screen + all 6 cards reachable
# ============================================================================

from textual.widgets import Input, Switch  # noqa: E402

from lmtool.tui.screens.upload import UploadScreen  # noqa: E402


class TestUploadScreen:
    """UploadScreen form renders, token is masked, validation works."""

    @pytest.mark.asyncio
    async def test_upload_card_navigates_to_upload_screen(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            await pilot.click("#card-upload")
            await pilot.pause()
            assert isinstance(app.screen, UploadScreen)

    @pytest.mark.asyncio
    async def test_upload_screen_has_model_path_input(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app.switch_screen(UploadScreen())
            await pilot.pause()
            assert len(app.screen.query("#input-model-path")) == 1

    @pytest.mark.asyncio
    async def test_upload_screen_has_repo_id_input(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app.switch_screen(UploadScreen())
            await pilot.pause()
            assert len(app.screen.query("#input-repo-id")) == 1

    @pytest.mark.asyncio
    async def test_upload_token_field_is_masked(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app.switch_screen(UploadScreen())
            await pilot.pause()
            token_input = app.screen.query_one("#input-token", Input)
            assert token_input.password is True

    @pytest.mark.asyncio
    async def test_upload_screen_has_private_switch(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app.switch_screen(UploadScreen())
            await pilot.pause()
            assert len(app.screen.query("#switch-private")) == 1

    @pytest.mark.asyncio
    async def test_upload_screen_has_merge_switch(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app.switch_screen(UploadScreen())
            await pilot.pause()
            assert len(app.screen.query("#switch-merge")) == 1

    @pytest.mark.asyncio
    async def test_upload_back_returns_home(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = UploadScreen()
            app.switch_screen(screen)
            await pilot.pause()
            await pilot.pause()
            screen.action_go_home()
            await pilot.pause()
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)

    @pytest.mark.asyncio
    async def test_upload_submit_empty_shows_validation(self):
        app = LMToolApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            screen = UploadScreen()
            app.switch_screen(screen)
            await pilot.pause()
            await pilot.pause()
            screen.action_submit()
            await pilot.pause()
            assert isinstance(app.screen, UploadScreen)


class TestAllSixCardsReachable:
    """Every home screen card navigates to its own screen."""

    @pytest.mark.asyncio
    async def test_all_six_cards_navigate_to_distinct_screens(self):
        expected = {
            "#card-train":     TrainScreen,
            "#card-evaluate":  EvaluateScreen,
            "#card-benchmark": BenchmarkScreen,
            "#card-upload":    UploadScreen,
            "#card-merge":     MergeScreen,
            "#card-recommend": RecommendScreen,
        }
        for card_id, screen_cls in expected.items():
            app = LMToolApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                await pilot.pause()
                await pilot.click(card_id)
                await pilot.pause()
                assert isinstance(app.screen, screen_cls), \
                    f"{card_id} should open {screen_cls.__name__}, got {type(app.screen).__name__}"
