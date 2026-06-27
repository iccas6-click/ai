from pill_recognition.settings import parse_bool


def test_parse_bool_defaults_when_missing():
    assert parse_bool(None, default=True) is True
    assert parse_bool(None, default=False) is False


def test_parse_bool_accepts_common_false_values():
    for value in ("0", "false", "no", "off"):
        assert parse_bool(value, default=True) is False


def test_parse_bool_treats_other_values_as_true():
    assert parse_bool("1", default=False) is True
    assert parse_bool("true", default=False) is True
