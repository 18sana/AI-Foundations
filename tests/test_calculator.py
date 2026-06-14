from src.calculator import calculator


def test_addition():
    assert calculator("2+2") == 4
