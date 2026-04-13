from pathlib import Path


def test_readme_exists_and_has_required_headings() -> None:
    readme_path = Path(__file__).resolve().parents[1] / "README.md"
    assert readme_path.exists()

    readme_text = readme_path.read_text(encoding="utf-8")

    required_headings = [
        "## Docker Compose quickstart",
        "## Pull Ollama models",
        "## Run tests",
        "### Test database setup (exact commands)",
    ]

    for heading in required_headings:
        assert heading in readme_text, f"Missing heading in README: {heading}"
