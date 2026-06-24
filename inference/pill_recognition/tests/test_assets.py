from pathlib import Path

from pill_recognition.assets import download_model_assets
from pill_recognition import settings as settings_module
from pill_recognition.settings import Settings


def test_download_model_assets_prefers_local_detector(tmp_path):
    checkpoint = tmp_path / "model.pth"
    classes = tmp_path / "pill.yaml"
    checkpoint.touch()
    classes.write_text("names:\n  0: pill\n", encoding="utf-8")

    assets = download_model_assets(
        Settings(
            detector_checkpoint=checkpoint,
            detector_class_names=classes,
            device="cpu",
        )
    )

    assert assets.checkpoint == Path(checkpoint)
    assert assets.class_names == Path(classes)


def test_settings_discovers_local_aihub_assets(tmp_path, monkeypatch):
    official_dir = tmp_path / "aihub_official_code" / "docker img" / "proj_pill"
    official_dir.mkdir(parents=True)
    weights = official_dir / "pill_resnet152_dataclass01_aug0.pt"
    mapping = official_dir / "pill_label_path_sharp_score.json"
    weights.touch()
    mapping.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)

    settings = Settings.from_env()

    assert settings.aihub_weights == weights
    assert settings.aihub_mapping == mapping


def test_settings_prefers_extracted_aihub_package(tmp_path, monkeypatch):
    data_dir = (
        tmp_path
        / "aihub_official_code"
        / "package"
        / "평가용 데이터셋"
        / "pill_data"
    )
    weights = data_dir / "proj_pill" / "pill_resnet152_dataclass01_aug0.pt"
    mapping = data_dir / "pill_data_croped" / "pill_label_path_sharp_score.json"
    weights.parent.mkdir(parents=True)
    mapping.parent.mkdir(parents=True)
    weights.touch()
    mapping.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(settings_module, "PROJECT_ROOT", tmp_path)

    settings = Settings.from_env()

    assert settings.aihub_weights == weights
    assert settings.aihub_mapping == mapping
