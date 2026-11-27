.PHONY: check format test build

check:
	uv run --locked ruff check .
	uv run --locked ruff format --check .
	uv run --locked pyright
	uv run --locked semgrep --config rules/semgrep --no-git-ignore src tests
	uv run --locked lint-imports --config rules/import-linter/contracts.ini
	uv run --locked vulture src --min-confidence 80
	uv run --locked deptry src
	uv run --locked pytest

format:
	uv run --locked ruff format .

test:
	uv run --locked pytest

build:
	uv build
