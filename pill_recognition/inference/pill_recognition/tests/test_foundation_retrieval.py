import torch

from pill_recognition.foundation_retrieval import (
    FoundationImageRetriever,
    encode_images,
)


def test_encode_images_accepts_dinov2_feature_dict():
    batch = torch.zeros((2, 3, 224, 224))

    class DictEncoder(torch.nn.Module):
        def forward(self, x):
            return {"x_norm_clstoken": torch.ones((x.shape[0], 384))}

    embeddings = encode_images(DictEncoder(), batch)

    assert embeddings.shape == (2, 384)


def test_foundation_retriever_aggregates_duplicate_reference_ids(monkeypatch, tmp_path):
    index_path = tmp_path / "foundation-index.pt"
    torch.save(
        {
            "version": 1,
            "index_mode": "reference",
            "encoder": "fake",
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
        "pill_recognition.foundation_retrieval.load_aihub_class_names",
        lambda path: {0: "K-000001", 1: "K-000002"},
    )
    monkeypatch.setattr(
        "pill_recognition.foundation_retrieval.load_aihub_product_master",
        lambda crop_root, pill_ids=None: {},
    )
    monkeypatch.setattr(
        "pill_recognition.foundation_retrieval.load_torchhub_encoder",
        lambda repo, model: torch.nn.Identity(),
    )

    mapping_path = tmp_path / "pill_label_path_sharp_score.json"
    mapping_path.write_text("{}", encoding="utf-8")
    retriever = FoundationImageRetriever(
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
    assert predictions[0][0].source == "foundation_image_retrieval"
