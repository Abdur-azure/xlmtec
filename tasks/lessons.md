## Pattern: Patch lazy-imported HF classes via the method, not the module
Trainer and DataCollatorForLanguageModeling are lazy-imported inside
_build_hf_trainer() in base.py. patch("xlmtec.trainers.base.Trainer")
fails with AttributeError — the attribute doesn't exist at module level.
Instead patch the method directly on the instance:
    with patch.object(trainer, "_build_hf_trainer", return_value=hf_instance):
        result = trainer.train(dataset)
Same rule applies to any lazy import inside a method: patch the method, not
the module attribute.

## Pattern: Mock the training mock return value needs metrics + global_step
When patching hf_trainer.train(), the return value must include:
    MagicMock(training_loss=0.42, metrics={"epoch": 1}, global_step=10)
Missing metrics or global_step causes TrainingResult construction to fail
with AttributeError. Always include all three fields.

## Pattern: Test data pipeline via DataPipeline.run() patch, not tokenizer mock
The data pipeline maps the tokenizer over batches internally. Mocking a
tokenizer to return the right dict shape is fragile and breaks when the
pipeline's batching strategy changes. Instead, patch DataPipeline.run()
to return a known Dataset. This tests wiring (quick_load → DataPipeline),
not tokenization internals — which is what a unit test should do.

## Pattern: pyproject.toml is declarative — no Python code runs during install, no
encoding bugs possible. Always use pyproject.toml for new projects.

## Pattern: Ship an audit script with generated file sets
When delivering many files across sessions, include audit_repo.py so the user
can instantly see which files are missing from their local repo without
manually comparing directory listings.

## Pattern: data/__init__.py must mirror CLI imports exactly
The CLI does `from ..data import quick_load` — if __init__.py doesn't
re-export quick_load the import fails silently at runtime not test time.
Always trace the full import chain from CLI → package → module when
verifying data pipeline wiring.

## Pattern: Deprecation shims beat deletion
Never delete v1 files users might be running. Replace the body with a
DeprecationWarning + sys.exit() that points to the v2 command.
This gives users a clear migration message instead of a FileNotFoundError.

## Pattern: Verify docs with term-presence checks, not just line counts
When rewriting docs, assert that key v2 identifiers (CLI commands, class names,
config keys) are present. A doc that builds without errors can still describe v1.
Always check content, not just structure.

## Pattern: Example data scripts must be zero-dependency
generate_sample_data.py uses only stdlib (json, random, pathlib).
Never import project modules or third-party libs in example generators —
they must run before pip install -e . succeeds.

## Pattern: Name sprints — makes changelog and reviews easier to navigate
Sprint names (e.g. "First Run", "Expand") give reviewers instant context
without reading every commit. Add to todo.md header and CHANGELOG.

## Pattern: Guard lora_config by method — not unconditionally
cli/main.py was calling .with_lora() for every method including full_finetuning.
Always check the method set before attaching method-specific config.
Pattern: define _LORA_METHODS = {LORA, QLORA, INSTRUCTION_TUNING} and gate on it.

## Pattern: CHANGELOG and audit_repo.py are first-class deliverables
After every sprint, update both. CHANGELOG tells humans what changed.
audit_repo.py tells Claude (and contributors) what files must exist.
Skipping these creates session drift — the next session starts blind.

## Pattern: Write test_cli_train.py for every new method added to CLI
When a new TrainingMethod is wired into the factory, add a CLI-level test
that invokes the train command with --method <new_method> and asserts
exit_code == 0. Mock the training stack — this is a wiring test, not a
training test. Catches missing lora_config guards before they hit production.

## Pattern: Use string concatenation not f-strings for multiline Panel content
When building Rich Panel content across multiple lines, avoid putting \n
inside f-strings — heredocs interpret them as literal newlines which breaks
the string literal. Instead build the string with concatenation:
    panel_text = "[header]\n" + f"field: {value}\n" ...
Or use a separate variable assigned before the Panel() call.

## Pattern: Implement lessons as commands, not just notes
lessons.md had "upload needs merge option" documented since Sprint 1.
It sat unbuilt for 4 sprints. During the drill, review lessons.md for
unactioned items and prioritise them as sprint tasks.

## Pattern: Install pytest-timeout whenever --timeout is used in CI
If ci.yml uses --timeout=N, pytest-timeout must be in the install step.
Missing it causes pytest to silently ignore the flag (no error, no timeout).
Always audit install step against all pytest flags used in the workflow.

## Pattern: tasks/CONTEXT.md sprint table drifts — update it every sprint
The sprint table in tasks/CONTEXT.md was frozen at Sprint 3 through Sprint 7.
Rule: when archiving a sprint in todo.md, also add a row to CONTEXT.md.

## Pattern: docs/index.md is a second source of truth — keep it in sync
Version number, test count, and component table in docs/index.md all drifted.
After each sprint, grep for hardcoded version strings and test counts and update them.

## Pattern: CI test path must match actual repo layout — verify against audit_repo.py
ci.yml had `pytest xlmtec/tests/` but tests live at `tests/` from repo root.
Rule: after writing ci.yml, cross-check every path against audit_repo.py REQUIRED_FILES.
If audit_repo lists "tests/test_config.py" (no xlmtec/ prefix), the CI path is `tests/`.

## Pattern: ruff config must use [tool.ruff.lint] not [tool.ruff]
Since ruff 0.1+, `select` and `ignore` belong under `[tool.ruff.lint]` in pyproject.toml.
Putting them under `[tool.ruff]` triggers a deprecation warning AND causes E902 errors
because ruff invalidates its own config. Always use the `lint` subsection.

## Pattern: PowerShell uses backtick for line continuation, not backslash
`pytest .\tests\test_cli_train.py -v \` — the trailing `\` on Windows is passed as a
literal argument (the drive root), causing pytest to collect the entire filesystem.
On Windows use backtick: `pytest .\tests\test_cli_train.py -v `
Or just keep commands on one line in the docs.

## Pattern: Tests in tests/ must use absolute imports, not relative
test_cli_train.py and test_merge.py used `from ..cli.main import app`.
Relative imports only work when the file is part of the package being traversed.
Tests in tests/ (repo root) are NOT inside xlmtec/, so `..` is invalid.
Always use absolute imports in test files: `from xlmtec.cli.main import app`.

## Pattern: conftest.py must not import torch at module level
The shared conftest.py had `import torch` + `torch.randn(10,10)` in mock_model.
This forces torch to be present at pytest collection time — all unit tests fail
to collect on machines without torch installed.
Fix: use pure MagicMock params with `param.numel.return_value = N` and
`param.requires_grad = True`. No real tensors anywhere in conftest.
Rule: conftest.py must be importable with only: pytest, unittest.mock, datasets, pathlib.

## Pattern: parameters() is called multiple times — use side_effect not return_value
If mock_model.parameters.return_value = iter([param]), the first call exhausts
the iterator. Subsequent calls return an empty iterator and parameter-count
logic silently returns 0.
Fix: model.parameters.side_effect = lambda: iter([param])
This returns a fresh iterator on every call.

## Pattern: patch() target must be where the name is USED, not where it's exported
patch("xlmtec.data.DataPipeline") patches __init__ — the pipeline functions
never see it. Correct: "xlmtec.data.pipeline.DataPipeline".
Rule: grep for the import in the file under test — that module path is your target.

## Pattern: Embed paths in YAML using .as_posix(), never raw str()
Windows paths contain backslashes (C:\Users\...). Inside YAML double-quoted
strings, \U, \s, \A etc. are invalid escape sequences → yaml.ScannerError.
Fix: always use path.as_posix() when writing paths into YAML content.
  WRONG: f'path: "{str(some_path)}"'
  RIGHT: f'path: "{some_path.as_posix()}"'
This applies in test fixtures and any code that builds YAML strings manually.

## Pattern: Update stale tests when a "unsupported" method becomes supported
test_unsupported_method_raises used DPO to trigger NotImplementedError.
DPO was added in Sprint 8 — the test became stale and now triggers
MissingConfigError instead. After adding any new TrainingMethod to the
factory, scan test_trainers.py for tests that depend on that method being
unsupported and update them to reflect the new reality.

## Pattern: Never call fixtures directly inside test bodies
Fixtures are injected by pytest as parameters — calling model_config() inside
a test method raises "Fixture called directly" error.
Fix: construct the object inline instead:
  WRONG: model_cfg = model_config()
  RIGHT: model_cfg = ModelConfig(name="gpt2", load_in_4bit=True)
Rule: if you need a fixture's value inside a helper/loop in a test, either
request it as a test parameter or construct the object directly inline.

## Pattern: Don't iterate all enum values when testing factory dispatch
TrainingMethod has 21 aspirational enum values; factory only implements 5.
Iterating all values in test_all_methods_are_handled hits NotImplementedError
for unimplemented methods (adalora, rlhf, etc.) and fails correctly but uselessly.
Fix: define _IMPLEMENTED = {LORA, QLORA, FULL_FINETUNING, INSTRUCTION_TUNING, DPO}
and only iterate that set. The test guards against regression on implemented methods,
not against missing future implementations.
Rule: when testing a factory, always scope iteration to the known-implemented set.

## Pattern: Embed paths in YAML using .as_posix(), never raw str()
Windows paths contain backslashes. Inside YAML double-quoted strings,
\U, \s, \A etc. are invalid escape sequences → yaml.ScannerError.
Fix: always use path.as_posix() when writing paths into YAML content.
  WRONG: f'path: "{str(some_path)}"'
  RIGHT: f'path: "{some_path.as_posix()}"'

## Pattern: Update stale tests when an "unsupported" method becomes supported
test_unsupported_method_raises used DPO to trigger NotImplementedError.
DPO was added in Sprint 8 — test became stale and triggered MissingConfigError.
After adding any new TrainingMethod: scan tests for ones that depend on it
being unsupported and update them.

## Pattern: Never call fixtures directly inside test bodies
model_config() inside a test method raises "Fixture called directly" error.
  WRONG: model_cfg = model_config()
  RIGHT: model_cfg = ModelConfig(name="gpt2", load_in_4bit=True)
Rule: if you need a fixture's value in a helper/loop, construct the object inline.

## Pattern: Don't iterate all enum values when testing factory dispatch
TrainingMethod has 21 aspirational enum values; factory only implements 5.
Iterating all values hits NotImplementedError for unimplemented methods.
Fix: define _IMPLEMENTED = {LORA, QLORA, FULL_FINETUNING, INSTRUCTION_TUNING, DPO}
and only iterate that set.
Rule: always scope iteration to the known-implemented set.

## Pattern: test_full_trainer.py used real torch tensors — violates no-real-tensor rule
test_full_trainer.py had `torch.nn.Parameter(torch.randn(4, 4))` in test bodies.
This imports torch at test time — breaks collection without torch installed.
Fix: replace with _make_param() helper returning a pure MagicMock:
    def _make_param(numel, requires_grad=True):
        param = MagicMock()
        param.numel.return_value = numel
        param.requires_grad = requires_grad
        return param
Rule: NEVER use real torch tensors in unit tests. MagicMock + numel.return_value only.

## Pattern: Apply core/types.py dataclass FIRST — before writing the trainer
Sprint 23 shipped ResponseDistillationTrainer before DistillationConfig was added
to core/types.py. Every test file that imported DistillationConfig failed collection
with ImportError, breaking 6 test files simultaneously.
Rule: Step 1 of any new trainer sprint is always core/types.py — add the config
dataclass and verify `python -c "from xlmtec.core.types import NewConfig"`
passes BEFORE writing the trainer file.
This is now step 0 in the trainer checklist, before even creating the trainer file.

## Pattern: pyproject.toml build-backend must be "setuptools.build_meta"
Using "setuptools.backends.legacy:build" causes BackendUnavailable on pip install -e .
The correct value is always:
  [build-system]
  requires = ["setuptools>=68", "wheel"]
  build-backend = "setuptools.build_meta"

## Pattern: MagicMock.to() returns a new mock — chain methods lose configured attributes
When a MagicMock has attributes set (e.g. t.config.num_hidden_layers = 24),
calling t.to(device) returns a fresh MagicMock, losing all configured state.
Fix: always add `t.to.return_value = t` when the code calls .to() on a mock object.
Same rule applies to any method that should "return self" (e.g. .eval(), .train(),
.cuda(), .cpu() — configure them explicitly if their return value is used).

## Pattern: Textual 8.x — push_screen in on_mount must be awaited
In Textual >=0.52, `push_screen()` returns a coroutine. Calling it synchronously
in `on_mount` silently does nothing — HomeScreen never mounts, all queries return 0.
Fix: make `on_mount` async and await the call:
    async def on_mount(self) -> None:
        await self.push_screen("home")

## Pattern: Textual 8.x — use switch_screen for action_go_home, not pop loop
`while len(screen_stack) > 1: pop_screen()` pops HomeScreen and leaves the
blank _default screen. Use `self.switch_screen("home")` instead — it replaces
the current screen with HomeScreen without touching the stack depth.

## Pattern: Textual tests — use app.screen.query() not app.query() for screen widgets
`app.query(Widget)` searches all screens including the blank _default screen.
Use `app.screen.query(Widget)` to scope queries to the active screen.
Also add a second `await pilot.pause()` after mount to let async screen pushes settle.

## Pattern: Textual tests — update "stays on HomeScreen" card tests when cards get wired
TestCommandCard click/enter tests assert `isinstance(app.screen, HomeScreen)`.
Once a card is wired to a real screen (Sprint 26 wired Train + Recommend),
those tests fail because the screen switches. Fix: use a still-stubbed card
(e.g. #card-evaluate, #card-benchmark) for the "stays on home" assertion.
Update these tests every sprint as more cards get wired.

## Pattern: Textual tests — add double pause() after switch_screen + button clicks
switch_screen() and button handlers are async. A single pause() isn't always
enough for the screen transition to settle. Use double pause() after any
switch_screen() call or button click that causes a screen change:
    app.switch_screen(SomeScreen())
    await pilot.pause()
    await pilot.pause()

## Pattern: Textual tests — call action_*() directly for validation tests
Clicking a submit button goes through Textual's event system with async
routing. In tests, if you need to assert the screen DIDN'T change after
validation failure, call the action method directly on the screen instance:
    screen.action_submit()   # bypasses event routing race
    await pilot.pause()
    assert isinstance(app.screen, SameScreen)

## Pattern: Textual tests — call action_go_home() directly for back/home navigation tests
pilot.click("#btn-back") goes through Textual's full async message dispatch.
Even with multiple pauses, switch_screen() may not complete within the test.
Consistent fix: call the action method directly on the screen instance:
    screen.action_go_home()   # immediate, synchronous, reliable in tests
    await pilot.pause()
    await pilot.pause()
Apply this to ALL back/home/cancel button tests across Sprint 26-28.

## Pattern: Textual 8.x Select — never set value= in constructor, use on_mount
In Textual 8.x, passing value= to Select() constructor raises
InvalidSelectValueError because options aren't processed yet at construction time.
Fix: omit value= from Select(), set it in on_mount after the widget is mounted:
    async def on_mount(self) -> None:
        self.query_one("#select-id", Select).value = "default_value"
Applies to ALL Select widgets in every TUI screen.

## Pattern: Textual test imports — import all widget types used in queries at top of test file
If a test does app.screen.query(Checkbox), Checkbox must be imported at the
top of the test file. Missing imports cause NameError at test collection time,
not a helpful assertion error. Add all widget types to the Sprint N import block
when writing tests that query them.

## Pattern: Textual Select options are (label, value) — label is displayed, value is returned
Textual Select options format: Iterable[tuple[str, SelectType]] where
  tuple[0] = label (displayed in dropdown)
  tuple[1] = value (what .value returns / what you set)
WRONG: [("float32", "float32 (default, safest)")]  ← value is the long string
RIGHT: [("float32  (default, safest)", "float32")]  ← label is long, value is short
Then on_mount: self.query_one("#sel", Select).value = "float32"  ← matches tuple[1]

## Pattern: Textual CSS — only use Textual's supported property subset
app.css caused 192 errors because standard CSS properties are not valid in Textual.
Textual supports its own subset. Properties to AVOID in app.css:
  - font-family, font-size, font-weight, font-style (use text-style instead)
  - box-shadow, text-shadow, filter, transform, transition, animation
  - display: flex/block/none (use layout: horizontal/vertical, display: none/block)
  - margin-top/right/bottom/left shorthand in some cases (use margin: T R B L)
  - scrollbar-color-hover (not supported — use scrollbar-color only)
Safe Textual properties: background, color, border, padding, margin, width,
height, layout, align, content-align, text-style, text-align, display, offset,
layer, opacity, scrollbar-color, grid-size, grid-gutter, dock.

## Pattern: Textual test stub cards — update every sprint as cards get wired
The "stub card stays on HomeScreen" tests must be updated each sprint.
Sprint 28 wired all 6 cards — no stubs remain. Replace the stub tests with:
  - test_click_card_does_not_crash: click any card, assert screen is not None
  - test_enter_on_focused_card: focus a specific card, assert its screen type

## Pattern: f-string format specs (,:d, etc.) fail on MagicMock — always cast to int/float first
f"{some_value:,}" calls __format__ with "," spec. If some_value is a MagicMock
(e.g. sum of mock.numel() returns), this raises TypeError.
Fix: always cast to int/float before formatting: f"{int(zeroed):,}"
Rule: any log or print line with a format spec must cast first if the value
could come from a mock in tests.

## Pattern: patch at source module, not at cli.main for inside-function imports
cli/main.py imports heavy deps inside the function body:
  def prune(...):
      from xlmtec.models.loader import load_model_and_tokenizer
This means the name is NOT on xlmtec.cli.main at module level.
Patching "xlmtec.cli.main.load_model_and_tokenizer" raises AttributeError.
Fix: patch at the SOURCE module where the function is defined:
  patch("xlmtec.models.loader.load_model_and_tokenizer")
  patch("xlmtec.trainers.structured_pruner.StructuredPruner")
This is already documented in lessons.md as the lazy-import patch pattern —
reinforced here because it recurred in test_prune.py.

## Pattern: int() on MagicMock returns a MagicMock, not a real int — cast at the source
`int(MagicMock())` does NOT reliably return a Python int. In CPython, MagicMock's
__int__ returns 1 (a real int), but under pytest/coverage hooks it can return
another MagicMock. Result: f"{int(mock):,}" still raises TypeError on __format__.
Fix: cast to int AT THE POINT where the value is produced, not at the f-string:
  total = 0
  for p in model.parameters():
      total += int(p.numel())    # cast here, not later
  return total                   # guaranteed plain int
And in the caller, pre-compute ALL formatted values before ANY f-string:
  zeroed_int = int(zeroed)       # guaranteed plain int
  f"zeroed {zeroed_int:,}"       # safe
Rule: any value that comes from a mock chain must be cast to int/float at
collection time, not at formatting time.

---

### Lesson: Value-threshold masking fails on uniform scores — use index-based ranking

**Symptom:** `test_apply_wanda_mask_global_zeroes_correct_fraction` fails with
`assert 32 <= 17` — all weights zeroed instead of 50%.

**Root cause:** `kthvalue(k).values` returns the k-th value (e.g. `1.0` for
uniform scores), then `scores > threshold` is `False` for every element when
all scores equal the threshold → entire matrix zeroed.

**Fix:** Replace value-threshold with index-based ranking:
```python
# WRONG — breaks on ties/uniform scores
threshold = scores.flatten().kthvalue(n_prune).values
mask = scores > threshold

# RIGHT — always zeroes exactly n_prune elements
_, sorted_indices = scores.flatten().sort()  # ascending
prune_flat_idx = sorted_indices[:n_prune]
mask = torch.ones(scores.numel(), dtype=torch.bool)
mask[prune_flat_idx] = False
mask = mask.reshape(scores.shape)
```

**Rule:** Any pruning/masking that uses a value threshold must use `>=` or
index-based selection to handle ties. Unit tests with uniform tensors (all
`torch.ones`) are the canonical regression test for this class of bug.

## Pattern: Split ML deps into optional extras — never put torch in mandatory dependencies
torch, transformers, peft, accelerate, bitsandbytes and textual are heavy optional deps.
Putting them in [project.dependencies] means `pip install xlmtec` pulls ~2.5GB.
Rule: keep mandatory deps lightweight (pydantic, typer, rich, yaml, tqdm, eval metrics).
Move GPU stack to [ml], TUI to [tui], DPO to [dpo], tests/lint to [dev], all to [full].
CI installs: `pip install -e ".[ml,tui,dev]"` for full test suite.

## Pattern: pytest-asyncio must be installed for TUI tests — it's not a transitive dep
test_tui.py uses asyncio_mode = "auto" via pytest-asyncio. If pytest-asyncio is not
installed the tests silently fail to collect (no error, no skip — just disappear).
Rule: always include pytest-asyncio in the [dev] extra AND verify it's in ci.yml install.

## Pattern: ci.yml install step must derive from pyproject.toml, not requirements.txt
requirements.txt is a legacy artifact and will always drift behind pyproject.toml.
Rule: delete requirements.txt. Use `pip install -e ".[extras]"` in ci.yml exclusively.
Cache key must hash pyproject.toml, not requirements.txt.

## Pattern: Lint job must install the package before running ruff
If ruff runs without the package installed, it cannot resolve project imports and
reports false-positive E402/F401 errors that fail CI. Fix: `pip install -e ".[dev]"` 
in the lint job before `ruff check .`.

## Pattern: Package rename requires 6 touch-points minimum
When renaming a Python package, ALL of these must change atomically:
1. The package folder name itself
2. pyproject.toml: name, scripts entrypoint, packages.find include, URLs
3. All Python imports: `from old_name.` → `from new_name.`
4. All test patch targets: `old_name.module.Class` → `new_name.module.Class`
5. All docs, README, CHANGELOG, CLAUDE.md
6. CLI command name in pyproject.toml [project.scripts]
Missing any one of these causes either import errors or the wrong command name
shipping to PyPI. Use a rename script (rename_to_xlmtec.py) to catch all cases.