import torch

from pill_recognition.retrieval import AIHubResNetRetriever


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
