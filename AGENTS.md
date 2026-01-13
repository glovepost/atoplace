# Repository Guidelines

## Project Structure & Module Organization
- `atoplace/` is the primary Python package.
- Core modules: `atoplace/board/`, `atoplace/placement/`, `atoplace/validation/`,
  `atoplace/dfm/`, `atoplace/nlp/`, `atoplace/routing/`, `atoplace/output/`.
- CLI entrypoint lives in `atoplace/cli.py`.
- Tests live in `tests/` (e.g., `tests/test_constraints.py`, `tests/test_nlp.py`).
- Planning docs live in `docs/` (see `docs/PRODUCT_PLAN.md`).
- Research notes and examples are in `research/` and `examples/` respectively.

## Build, Test, and Development Commands
- Install dev dependencies: `pip install -e ".[dev]"`
- Run tests: `pytest`
- Format code: `black atoplace/`
- Lint: `ruff check atoplace/`
- Type check: `mypy atoplace/`
- Run CLI: `atoplace place board.kicad_pcb`

## Coding Style & Naming Conventions
- Python: 4-space indentation, `snake_case` for functions/variables,
  `PascalCase` for classes.
- Formatting is enforced by Black (`line-length = 88` in `pyproject.toml`).
- Linting uses Ruff with import sorting enabled; fix lint warnings before merging.
- Keep public APIs explicit by updating exports in `atoplace/__init__.py`.
- Prefer module-level docstrings for new modules, mirroring existing files.

## Testing Guidelines
- Framework: `pytest` with `tests/` as the root.
- Test file pattern: `test_*.py` (see `pyproject.toml` settings).
- Focus on deterministic behaviors (constraints, parsing, board geometry).
- Add new tests alongside the subsystem being changed.

## Commit & Pull Request Guidelines
- No existing git history in this workspace, so follow a clear imperative style:
  `Add constraint solver scoring`, `Fix KiCad adapter path handling`.
- PRs should include: scope summary, test command(s) run, and any constraints
  or KiCad version assumptions.

## Environment & Configuration Tips
- KiCad’s Python API (`pcbnew`) is required for real board IO; run the CLI with
  KiCad’s bundled Python when needed.
- On macOS, KiCad’s Python requires a logged-in GUI session on the main display
  (headless/SSH runs will fail with the screen access error). Use the
  `scripts/atoplace-kicad` wrapper for convenience.
