from pill_recognition.vision_providers.gemini import parse_batch_observations


def test_parse_batch_observations_maps_results_by_pill_index():
    observations = parse_batch_observations(
        {
            "results": [
                {
                    "pill_index": 2,
                    "candidates": [
                        {
                            "product_name": "두번째정",
                            "ingredient": "성분B",
                            "caution_points": ["주의B"],
                            "confidence": 0.7,
                        }
                    ],
                    "confidence": 0.7,
                },
                {
                    "pill_index": 1,
                    "candidates": [
                        {
                            "product_name": "첫번째정",
                            "ingredient": "성분A",
                            "caution_points": ["주의A"],
                            "confidence": 0.8,
                        }
                    ],
                    "confidence": 0.8,
                },
            ]
        },
        expected_count=2,
        provider_name="gemini",
        model="test-model",
    )

    assert observations[0].product_candidates[0].product_name == "첫번째정"
    assert observations[1].product_candidates[0].ingredient == "성분B"
    assert observations[1].product_candidates[0].caution_points == ["주의B"]


def test_parse_batch_observations_returns_placeholder_for_missing_index():
    observations = parse_batch_observations(
        {"results": [{"pill_index": 1, "candidates": [], "confidence": 0}]},
        expected_count=2,
        provider_name="gemini",
        model="test-model",
    )

    assert observations[1].confidence == 0.0
    assert "did not include" in observations[1].notes
