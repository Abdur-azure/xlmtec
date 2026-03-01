## Pattern: Patch lazy-imported HF classes via the method, not the module
Trainer and DataCollatorForLanguageModeling are lazy-imported inside
_build_hf_trainer() in base.py. patch("finetune_cli.trainers.base.Trainer")
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
ci.yml had `pytest finetune_cli/tests/` but tests live at `tests/` from repo root.
Rule: after writing ci.yml, cross-check every path against audit_repo.py REQUIRED_FILES.
If audit_repo lists "tests/test_config.py" (no finetune_cli/ prefix), the CI path is `tests/`.

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
Tests in tests/ (repo root) are NOT inside finetune_cli/, so `..` is invalid.
Always use absolute imports in test files: `from finetune_cli.cli.main import app`.

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
patch("finetune_cli.data.DataPipeline") patches __init__ — the pipeline functions
never see it. Correct: "finetune_cli.data.pipeline.DataPipeline".
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