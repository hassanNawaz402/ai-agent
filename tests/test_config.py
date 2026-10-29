from app.config import load_config


def test_config_loads_and_has_dev_profile():
    cfg = load_config()
    assert cfg.active_env in ("dev", "prod")
    assert cfg.active_profile.model
    assert cfg.active_profile.provider in {
        "local_qwen", "openai", "anthropic", "google", "mistral",
        "cohere", "azure_openai", "aws_bedrock", "groq", "together", "xai",
    }
    assert cfg.conda_env_name == f"{cfg.active_env}_advantage_removals"
    assert cfg.database.name
    assert cfg.extract_topics  # non-empty allow-list


def test_check_required_env_vars_names_only():
    cfg = load_config()
    # Just assert the method runs and returns a list of strings.
    missing = cfg.check_required_env_vars()
    assert isinstance(missing, list)
    assert all(isinstance(x, str) for x in missing)
