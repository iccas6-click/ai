from pill_recognition.settings import Settings, parse_bool


def test_parse_bool_defaults_when_missing():
    assert parse_bool(None, default=True) is True
    assert parse_bool(None, default=False) is False


def test_parse_bool_accepts_common_false_values():
    for value in ("0", "false", "no", "off", ""):
        assert parse_bool(value, default=True) is False


def test_parse_bool_treats_other_values_as_true():
    assert parse_bool("1", default=False) is True
    assert parse_bool("true", default=False) is True


def test_settings_defaults_to_service_top3(monkeypatch, tmp_path):
    monkeypatch.delenv("PILL_TOP_K", raising=False)
    monkeypatch.setattr("pill_recognition.settings.PROJECT_ROOT", tmp_path)

    settings = Settings.from_env()

    assert settings.top_k == 3
    assert settings.max_batch_crops == 12
    assert settings.candidate_min_score == 70
    assert settings.candidate_ambiguity_margin == 3


def test_settings_reads_candidate_policy_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("PILL_TOP_K", "4")
    monkeypatch.setenv("PILL_MAX_BATCH_CROPS", "8")
    monkeypatch.setenv("PILL_CANDIDATE_MIN_SCORE", "80")
    monkeypatch.setenv("PILL_CANDIDATE_AMBIGUITY_MARGIN", "5")
    monkeypatch.setattr("pill_recognition.settings.PROJECT_ROOT", tmp_path)

    settings = Settings.from_env()

    assert settings.top_k == 4
    assert settings.max_batch_crops == 8
    assert settings.candidate_min_score == 80
    assert settings.candidate_ambiguity_margin == 5
