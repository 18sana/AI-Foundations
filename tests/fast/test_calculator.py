from src.calculator import calculator


def test_addition():
    assert calculator("2+2") == 4


def test_multiplication():
    assert calculator("5*6") == 30


def test_parentheses():
    assert calculator("(5+5)*2") == 20

# def test_failure_demo():
#     assert calculator("2+2") == 5