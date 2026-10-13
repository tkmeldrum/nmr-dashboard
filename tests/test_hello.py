from nmrdash.pipeline import hello


def test_hello():
    assert hello()["sum"] == 45.0
