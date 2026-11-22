.PHONY: check format test build

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright
	uv run semgrep --config rules/semgrep --no-git-ignore src tests
	uv run lint-imports --config rules/import-linter/contracts.ini
	uv run vulture src --min-confidence 80
	uv run deptry src
	uv run pytest

format:
	uv run ruff format .

test:
	uv run pytest

build:
	uv build
