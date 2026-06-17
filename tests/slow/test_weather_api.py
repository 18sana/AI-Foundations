import requests


def test_open_meteo_api():

    url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=51.5074"
        "&longitude=-0.1278"
        "&current=temperature_2m"
    )

    response = requests.get(
        url,
        timeout=10
    )

    assert response.status_code == 200

    data = response.json()

    assert "current" in data