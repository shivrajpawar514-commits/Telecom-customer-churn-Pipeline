import os
import re
import subprocess
import pytest


def test_no_hardcoded_aws_keys_in_source():
    """Verify that no AWS access keys (AKIA...) are hardcoded in source files."""
    aws_key_pattern = re.compile(r"AKIA[0-9A-Z]{16}")
    exclude_dirs = {".git", ".venv", "venv", "__pycache__", ".pytest_cache"}

    violations = []
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for f in files:
            if f.endswith((".py", ".yaml", ".yml", ".json", ".md", ".sh", "Dockerfile")):
                file_path = os.path.join(root, f)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
                        content = fh.read()
                        matches = aws_key_pattern.findall(content)
                        if matches:
                            violations.append((file_path, matches))
                except Exception:
                    pass

    assert len(violations) == 0, f"Found hardcoded AWS keys in: {violations}"


def test_env_file_not_tracked_in_git():
    """Verify that active .env file is not tracked in git version control."""
    res = subprocess.run(["git", "ls-files", ".env"], capture_output=True, text=True)
    assert res.stdout.strip() == "", ".env file must NOT be tracked in git!"


def test_gitignore_contains_sensitive_patterns():
    """Verify that .gitignore excludes environment variables and key files."""
    assert os.path.exists(".gitignore"), ".gitignore must exist"
    with open(".gitignore", "r") as f:
        content = f.read()

    assert ".env" in content
    assert "*.pkl" in content or "artifacts/" in content


def test_dockerignore_contains_sensitive_patterns():
    """Verify that .dockerignore excludes .env and secret files from docker context."""
    assert os.path.exists(".dockerignore"), ".dockerignore must exist"
    with open(".dockerignore", "r") as f:
        content = f.read()

    assert ".env" in content


def test_env_example_uses_placeholders():
    """Verify that .env.example contains dummy placeholders, not real credentials."""
    assert os.path.exists(".env.example"), ".env.example must exist"
    with open(".env.example", "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("AWS_ACCESS_KEY_ID=") or line.startswith("AWS_SECRET_ACCESS_KEY="):
            val = line.split("=", 1)[1]
            assert val in ["changeme", "your-access-key-id", "your-secret-access-key", ""], (
                f"Real credentials suspected in .env.example: {line}"
            )
