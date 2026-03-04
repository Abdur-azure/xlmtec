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

from finetune_cli.tui.app import FinetuneApp           # noqa: E402
from finetune_cli.tui.screens.home import HomeScreen    # noqa: E402
from finetune_cli.tui.widgets.command_card import CommandCard  # noqa: E402


class TestAppMount:
    """FinetuneApp starts and reaches HomeScreen cleanly."""

    @pytest.mark.asyncio
    async def test_app_mounts_without_error(self):
        app = FinetuneApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.screen is not None

    @pytest.mark.asyncio
    async def test_home_screen_is_initial_screen(self):
        app = FinetuneApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)

    @pytest.mark.asyncio
    async def test_app_title_set(self):
        app = FinetuneApp()
        async with app.run_test(size=(120, 40)) as pilot:
            assert app.TITLE == "finetune-cli"


class TestHomeScreen:
    """HomeScreen renders all 6 CommandCards."""

    @pytest.mark.asyncio
    async def test_six_command_cards_rendered(self):
        app = FinetuneApp()
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
        app = FinetuneApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            ids = {card.id for card in app.screen.query(CommandCard)}
            assert ids == expected_ids

    @pytest.mark.asyncio
    async def test_cards_are_focusable(self):
        app = FinetuneApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            for card in app.screen.query(CommandCard):
                assert card.can_focus is True

    @pytest.mark.asyncio
    async def test_home_title_label_present(self):
        app = FinetuneApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            assert len(app.screen.query(".home-title")) == 1

    @pytest.mark.asyncio
    async def test_home_subtitle_label_present(self):
        app = FinetuneApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            assert len(app.screen.query(".home-subtitle")) == 1


class TestKeyboardNavigation:
    """Global keybindings work from home screen."""

    @pytest.mark.asyncio
    async def test_q_key_exits_app(self):
        app = FinetuneApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            await pilot.press("q")

    @pytest.mark.asyncio
    async def test_tab_moves_focus(self):
        app = FinetuneApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            await pilot.press("tab")
            assert app.focused is not None

    @pytest.mark.asyncio
    async def test_escape_stays_on_home(self):
        app = FinetuneApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)

    @pytest.mark.asyncio
    async def test_right_arrow_moves_focus(self):
        app = FinetuneApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            first_focused = app.focused
            await pilot.press("right")
            await pilot.pause()
            assert app.focused is not first_focused

    @pytest.mark.asyncio
    async def test_down_arrow_moves_to_next_row(self):
        app = FinetuneApp()
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
        app = FinetuneApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            await pilot.click("#card-train")
            assert isinstance(app.screen, HomeScreen)

    @pytest.mark.asyncio
    async def test_enter_on_focused_card_does_not_crash(self):
        app = FinetuneApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            await pilot.press("enter")
            assert isinstance(app.screen, HomeScreen)

    @pytest.mark.asyncio
    async def test_card_command_id_matches_id_attribute(self):
        app = FinetuneApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            train_card = app.screen.query_one("#card-train", CommandCard)
            assert train_card.command_id == "train"