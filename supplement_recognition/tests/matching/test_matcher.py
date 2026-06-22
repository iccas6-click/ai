from unittest.mock import patch

from src.matching.matcher import match_and_enrich
from src.matching.mfds_client import MfdsProduct
from src.schema.result import SupplementProduct


def _base_product() -> SupplementProduct:
    return SupplementProduct(product_name="오메가3", functional_ingredients=[], confidence=0.0)


def test_match_found_sets_product_code():
    mock_mfds = MfdsProduct(
        product_code="MFDS_001",
        product_name="오메가3 1000mg",
        functional_ingredients=["EPA 및 DHA 함유 유지"],
    )
    with patch("src.matching.matcher.search_product", return_value=mock_mfds):
        result = match_and_enrich(_base_product())

    assert result.product_code == "MFDS_001"
    assert result.confidence == 0.85


def test_match_not_found_returns_low_confidence():
    with patch("src.matching.matcher.search_product", return_value=None):
        result = match_and_enrich(_base_product())

    assert result.product_code is None
    assert result.confidence == 0.5
