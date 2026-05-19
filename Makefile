.PHONY: install dev test lint format serve serve-dev eval clean

install:
	uv sync

dev:
	uv sync --group dev

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check .
	uv run mypy .

format:
	uv run ruff format .
	uv run ruff check --fix .

eval:
	PYTHONPATH=agents:libs:. uv run python -m evals.routing_eval

serve:
	uv run langgraph dev --config langgraph.json

serve-dev:
	uv run langgraph dev --config langgraph.dev.json

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +
