.PHONY: install test lint format clean
install:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ty check .

format:
	uv run ruff format .
	uv run ty format .

docs:
    uv run zensical serve

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache
