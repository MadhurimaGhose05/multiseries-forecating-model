from fastapi.testclient import TestClient
from app.main import app
client=TestClient(app)

def test_health(): assert client.get("/health").json()["status"]=="ok"
def test_bad_extension():
    r=client.post("/forecast",files={"file":("x.txt",b"a,b")},data={"horizon":24})
    assert r.status_code==400
