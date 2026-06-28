from pill_recognition.product_db import (
    ProductSearchQuery,
    imprint_variants,
    query_imprint_variants,
    search_products,
)
from pill_recognition_legacy.aihub_classifier import AIHubProductInfo


def test_search_products_ranks_imprint_shape_color_match():
    products = {
        "K-000001": AIHubProductInfo(
            pill_id="K-000001",
            product_name="대화와르파린나트륨정",
            ingredient="와르파린나트륨",
            print_front="W분할선2",
            drug_shape="원형",
            color_class1="하양",
        )
    }

    results = search_products(
        products,
        ProductSearchQuery(imprint="W2", shape="원형", color="하양"),
    )

    assert results[0]["pill_id"] == "K-000001"
    assert results[0]["score"] == 170


def test_search_products_matches_ingredient_text():
    products = {
        "K-000001": AIHubProductInfo(
            pill_id="K-000001",
            product_name="조이렉스정",
            ingredient="아시클로버",
        )
    }

    assert search_products(products, ProductSearchQuery(text="아시클로버"))[0][
        "pill_id"
    ] == "K-000001"


def test_imprint_variants_remove_split_line_word():
    assert "D5" in imprint_variants("D분할선5")


def test_query_imprint_variants_splits_front_and_back_terms():
    assert {"W2", "CKD"}.issubset(query_imprint_variants("W2 CKD"))
