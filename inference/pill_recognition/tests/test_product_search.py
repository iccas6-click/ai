from pill_recognition.aihub_classifier import AIHubProductInfo
from pill_recognition.product_search import (
    ProductSearchQuery,
    imprint_variants,
    search_products,
)


def test_search_products_ranks_exact_imprint_first():
    products = {
        "K-000001": AIHubProductInfo(
            pill_id="K-000001",
            product_name="테스트정",
            print_front="CKD",
            drug_shape="원형",
            color_class1="하양",
        ),
        "K-000002": AIHubProductInfo(
            pill_id="K-000002",
            product_name="다른정",
            print_front="CK",
            drug_shape="원형",
            color_class1="하양",
        ),
    }

    results = search_products(
        products,
        ProductSearchQuery(imprint="ckd", shape="원형", color="하양"),
    )

    assert results[0]["pill_id"] == "K-000001"
    assert results[0]["matched"] == "각인 exact, 모양, 색"


def test_search_products_matches_text_fields():
    products = {
        "K-000001": AIHubProductInfo(
            pill_id="K-000001",
            product_name="트윈스타정",
            company="한국베링거인겔하임",
            item_seq="201005083",
            ingredient="텔미사르탄|암로디핀베실산염",
        )
    }

    assert search_products(products, ProductSearchQuery(text="암로디핀"))[0][
        "pill_id"
    ] == "K-000001"
    assert search_products(products, ProductSearchQuery(text="201005083"))[0][
        "pill_id"
    ] == "K-000001"


def test_imprint_variants_remove_split_line_word():
    assert "D5" in imprint_variants("D분할선5")
