import torch

from pill_recognition.retrieval import AIHubResNetRetriever
from pill_recognition_legacy.aihub_classifier import AIHubProductInfo


def test_reference_index_aggregates_duplicate_pill_ids(monkeypatch, tmp_path):
    index_path = tmp_path / "index.pt"
    torch.save(
        {
            "version": 1,
            "index_mode": "reference",
            "pill_ids": ["K-000001", "K-000001", "K-000002"],
            "embeddings": torch.tensor(
                [
                    [1.0, 0.0],
                    [0.9, 0.1],
                    [0.0, 1.0],
                ]
            ),
        },
        index_path,
    )

    monkeypatch.setattr(
        "pill_recognition.retrieval.load_aihub_class_names",
        lambda path: {0: "K-000001", 1: "K-000002"},
    )
    monkeypatch.setattr(
        "pill_recognition.retrieval.load_aihub_product_master",
        lambda crop_root, pill_ids=None: {},
    )
    monkeypatch.setattr(
        "pill_recognition.retrieval.load_aihub_resnet_encoder",
        lambda weights_path: torch.nn.Identity(),
    )

    weights_path = tmp_path / "weights.pt"
    mapping_path = tmp_path / "pill_label_path_sharp_score.json"
    weights_path.touch()
    mapping_path.write_text("{}", encoding="utf-8")
    retriever = AIHubResNetRetriever(
        weights_path,
        mapping_path,
        index_path,
        device="cpu",
        rotation_tta=False,
    )
    monkeypatch.setattr(
        retriever,
        "embed_crops",
        lambda crops: torch.tensor([[1.0, 0.0]]),
    )

    predictions = retriever.predict_batch([object()], top_k=2)

    assert [candidate.pill_id for candidate in predictions[0]] == [
        "K-000001",
        "K-000002",
    ]
    assert predictions[0][0].score == 100.0


def test_predict_batch_can_search_only_allowed_pill_ids(monkeypatch, tmp_path):
    index_path = tmp_path / "index.pt"
    torch.save(
        {
            "version": 1,
            "index_mode": "reference",
            "pill_ids": ["K-OUTSIDE", "K-ALLOWED", "K-ALLOWED"],
            "embeddings": torch.tensor(
                [
                    [1.0, 0.0],
                    [0.8, 0.2],
                    [0.7, 0.3],
                ]
            ),
        },
        index_path,
    )

    monkeypatch.setattr(
        "pill_recognition.retrieval.load_aihub_class_names",
        lambda path: {0: "K-OUTSIDE", 1: "K-ALLOWED"},
    )
    monkeypatch.setattr(
        "pill_recognition.retrieval.load_aihub_product_master",
        lambda crop_root, pill_ids=None: {},
    )
    monkeypatch.setattr(
        "pill_recognition.retrieval.load_aihub_resnet_encoder",
        lambda weights_path: torch.nn.Identity(),
    )

    weights_path = tmp_path / "weights.pt"
    mapping_path = tmp_path / "pill_label_path_sharp_score.json"
    weights_path.touch()
    mapping_path.write_text("{}", encoding="utf-8")
    retriever = AIHubResNetRetriever(
        weights_path,
        mapping_path,
        index_path,
        device="cpu",
        rotation_tta=False,
    )
    monkeypatch.setattr(
        retriever,
        "embed_crops",
        lambda crops: torch.tensor([[1.0, 0.0]]),
    )

    unrestricted = retriever.predict_batch([object()], top_k=1)
    scoped = retriever.predict_batch(
        [object()],
        top_k=1,
        allowed_pill_ids={"K-ALLOWED"},
    )

    assert unrestricted[0][0].pill_id == "K-OUTSIDE"
    assert scoped[0][0].pill_id == "K-ALLOWED"


def test_predict_batch_returns_empty_when_allowed_scope_has_no_index_match(
    monkeypatch,
    tmp_path,
):
    index_path = tmp_path / "index.pt"
    torch.save(
        {
            "version": 1,
            "index_mode": "prototype",
            "pill_ids": ["K-000001"],
            "embeddings": torch.tensor([[1.0, 0.0]]),
        },
        index_path,
    )

    monkeypatch.setattr(
        "pill_recognition.retrieval.load_aihub_class_names",
        lambda path: {0: "K-000001"},
    )
    monkeypatch.setattr(
        "pill_recognition.retrieval.load_aihub_product_master",
        lambda crop_root, pill_ids=None: {},
    )
    monkeypatch.setattr(
        "pill_recognition.retrieval.load_aihub_resnet_encoder",
        lambda weights_path: torch.nn.Identity(),
    )

    weights_path = tmp_path / "weights.pt"
    mapping_path = tmp_path / "pill_label_path_sharp_score.json"
    weights_path.touch()
    mapping_path.write_text("{}", encoding="utf-8")
    retriever = AIHubResNetRetriever(
        weights_path,
        mapping_path,
        index_path,
        device="cpu",
        rotation_tta=False,
    )
    monkeypatch.setattr(
        retriever,
        "embed_crops",
        lambda crops: torch.tensor([[1.0, 0.0]]),
    )

    predictions = retriever.predict_batch(
        [object(), object()],
        top_k=3,
        allowed_pill_ids={"K-MISSING"},
    )

    assert predictions == [[], []]


def test_metadata_rerank_can_promote_matching_visual_candidate(monkeypatch, tmp_path):
    index_path = tmp_path / "index.pt"
    torch.save(
        {
            "version": 1,
            "index_mode": "prototype",
            "pill_ids": ["K-BLUE", "K-WHITE"],
            "embeddings": torch.tensor(
                [
                    [1.0, 0.0],
                    [0.995, 0.0999],
                ]
            ),
        },
        index_path,
    )

    monkeypatch.setattr(
        "pill_recognition.retrieval.load_aihub_class_names",
        lambda path: {0: "K-BLUE", 1: "K-WHITE"},
    )
    monkeypatch.setattr(
        "pill_recognition.retrieval.load_aihub_product_master",
        lambda crop_root, pill_ids=None: {
            "K-BLUE": AIHubProductInfo(
                pill_id="K-BLUE",
                product_name="파란약",
                drug_shape="장방형",
                color_class1="파랑",
            ),
            "K-WHITE": AIHubProductInfo(
                pill_id="K-WHITE",
                product_name="하얀약",
                drug_shape="원형",
                color_class1="하양",
            ),
        },
    )
    monkeypatch.setattr(
        "pill_recognition.retrieval.load_aihub_resnet_encoder",
        lambda weights_path: torch.nn.Identity(),
    )

    weights_path = tmp_path / "weights.pt"
    mapping_path = tmp_path / "pill_label_path_sharp_score.json"
    weights_path.touch()
    mapping_path.write_text("{}", encoding="utf-8")
    retriever = AIHubResNetRetriever(
        weights_path,
        mapping_path,
        index_path,
        device="cpu",
        rotation_tta=False,
        metadata_rerank=True,
    )
    monkeypatch.setattr(
        retriever,
        "embed_crops",
        lambda crops: torch.tensor([[1.0, 0.0]]),
    )

    white_round_crop = torch.full((80, 80, 3), 245, dtype=torch.uint8).numpy()
    predictions = retriever.predict_batch([white_round_crop], top_k=1)

    assert predictions[0][0].pill_id == "K-WHITE"
    assert predictions[0][0].matched == (
        "AIHub ResNet embedding similarity + metadata rerank (color=하양, shape=원형)"
    )
