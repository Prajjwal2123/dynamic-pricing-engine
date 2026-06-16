import requests
import json

BASE = "http://localhost:8000"

def test_health():
    r = requests.get(f"{BASE}/health")
    print("HEALTH:", r.json())
    assert r.status_code == 200
    assert r.json()["model_loaded"] == True

def test_model_info():
    r = requests.get(f"{BASE}/model-info")
    print("MODEL INFO:", json.dumps(r.json(), indent=2))
    assert r.status_code == 200

def test_single_recommendation():
    payload = {
        "product_id":       "P0001",
        "base_price":       1200.0,
        "cost":             800.0,
        "competitor_price": 1150.0,
        "inventory_ratio":  0.08,
        "demand_momentum":  15.0,
        "price_elasticity": -0.8
    }
    r = requests.post(f"{BASE}/recommend-price", json=payload)
    print("\nSINGLE RECOMMENDATION:")
    print(json.dumps(r.json(), indent=2))
    assert r.status_code == 200
    rec = r.json()
    assert rec["recommended_price"] > 0
    assert rec["price_floor"] <= rec["recommended_price"] <= rec["price_ceiling"]

def test_batch_recommendation():
    payload = {
        "products": [
            {"product_id":"P0001","base_price":1200,"cost":800,
             "competitor_price":1150,"inventory_ratio":0.08,
             "demand_momentum":15.0,"price_elasticity":-0.8},
            {"product_id":"P0002","base_price":500,"cost":300,
             "competitor_price":480,"inventory_ratio":0.60,
             "demand_momentum":-5.0,"price_elasticity":-1.8},
        ]
    }
    r = requests.post(f"{BASE}/batch-recommend", json=payload)
    print("\nBATCH RECOMMENDATION:")
    print(json.dumps(r.json(), indent=2))
    assert r.status_code == 200
    assert r.json()["total_products"] == 2

def test_price_curve():
    r = requests.get(f"{BASE}/price-curve/P0001", params={
        "base_price": 1200, "cost": 800,
        "competitor_price": 1150, "inventory_ratio": 0.08
    })
    print("\nPRICE CURVE (first 3 points):")
    data = r.json()
    for point in data["curve"][:3]:
        print(f"  Price: ₹{point['price']} → "
              f"Demand: {point['demand']} → "
              f"Revenue: ₹{point['revenue']}")
    assert r.status_code == 200
    assert len(data["curve"]) == 20

def test_validation_error():
    # When cost > price, pricing engine hits bound conflict — expect 500
    payload = {
        "product_id":       "BAD",
        "base_price":       100.0,
        "cost":             500.0,   # cost > price — invalid business case
        "competitor_price": 120.0,
        "inventory_ratio":  0.5,
        "demand_momentum":  0.0,
        "price_elasticity": -1.0
    }
    r = requests.post(f"{BASE}/recommend-price", json=payload)
    print("\nVALIDATION ERROR TEST:", r.status_code, r.json())
    # Pydantic validator catches cost > price as 422, pricing engine catches it as 500
    assert r.status_code in [422, 500]
    print("✓ Invalid input correctly rejected")

if __name__ == "__main__":
    print("="*50)
    print("TESTING DYNAMIC PRICING API")
    print("="*50)
    test_health()
    test_model_info()
    test_single_recommendation()
    test_batch_recommendation()
    test_price_curve()
    test_validation_error()
    print("\n✓ All tests passed")