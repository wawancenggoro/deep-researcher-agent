uv run pytest tests/deep_researcher_agent/agents/nvidia_poem -q
uv run ruff check src/deep_researcher_agent/agents/nvidia_poem tests/deep_researcher_agent/agents/nvidia_poem
uv run ruff format --check src/deep_researcher_agent/agents/nvidia_poem tests/deep_researcher_agent/agents/nvidia_poem
dotenv -f deploy/.env run nat run --config_file configs/config_cli_nvidia_poem.yml --input "make me a poem"
