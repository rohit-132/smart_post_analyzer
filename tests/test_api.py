from app import app

def test_predict():
    client = app.test_client()

    response = client.post('/api/predict', json={
        "Post_Type": "Reel",
        "Likes": 100,
        "Comments": 10,
        "Shares": 5,
        "Saves": 2,
        "Caption": "great post"
    })

    assert response.status_code == 200