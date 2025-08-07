import pytest
from flask import Flask
import json
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_route(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Iris' in response.data or b'iris' in response.data

def test_predict_api_valid(client):
    # Example valid input
    payload = {
        "data": {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    }
    response = client.post('/predict_api',
                           data=json.dumps(payload),
                           content_type='application/json')
    assert response.status_code == 200
    data = response.get_json()
    assert 'prediction' in data

def test_predict_api_invalid(client):
    # Missing required fields
    payload = {"data": {"sepal_length": 5.1}}
    response = client.post('/predict_api',
                           data=json.dumps(payload),
                           content_type='application/json')
    assert response.status_code == 200
    data = response.get_json()
    assert 'error' in data

def test_predict_form_valid(client):
    form_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post('/predict', data=form_data)
    assert response.status_code == 200
    assert b'IRIS species' in response.data or b'iris species' in response.data

def test_predict_form_invalid(client):
    # Missing one field
    form_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4
    }
    response = client.post('/predict', data=form_data)
    assert response.status_code == 200
    assert b'Error' in response.data

