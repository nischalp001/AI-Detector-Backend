import requests
import time
import json

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 20  # seconds
RETRIES = 3


# ---------------------------------------
# Helper: Robust Request Handler
# ---------------------------------------
def safe_request(method, url, **kwargs):
    """Make HTTP requests with retries, timeouts, and clean error handling."""
    for attempt in range(1, RETRIES + 1):
        try:
            resp = requests.request(method, url, timeout=TIMEOUT, **kwargs)
            resp.raise_for_status()
            return resp
        except Exception as e:
            print(f"[Attempt {attempt}/{RETRIES}] Error:", e)
            if attempt < RETRIES:
                print("Retrying in 2 sec...")
                time.sleep(2)
    print("❌ Request failed after retries.")
    return None


# ---------------------------------------
# 1️⃣ Test health endpoint
# ---------------------------------------
def test_health():
    url = f"{BASE_URL}/health"
    print("\n🌱 Testing /health ...")
    resp = safe_request("GET", url)
    if resp:
        print("Health:", resp.json())


# ---------------------------------------
# 2️⃣ Test single prediction
# ---------------------------------------
def test_single_predict(text: str):
    url = f"{BASE_URL}/predict"
    payload = {"text": text}

    print("\n🧪 Testing /predict ...")
    resp = safe_request("POST", url, json=payload)

    if resp:
        try:
            print(json.dumps(resp.json(), indent=4, ensure_ascii=False))
        except:
            print(resp.text)



# ---------------------------------------
# MAIN TEST RUNNER
# ---------------------------------------
if __name__ == "__main__":
    print("🔍 Starting API Tests...")

    # Health check
    test_health()

    # Single test — long multi-paragraph text
    sample_text = """
    
   ## The Dog: A Mirror to Humanity

Although there are advantages, the implementation of IaC does have a number of challenges that organisations and teams should deal with. One of the biggest obstacles is the learning curve related to acquiring the IaC tools, e.g. Terraform, Ansible, or AWS CloudFormation, particularly to a student or a novice DevOps professional, who may also require a good understanding of cloud architecture. The other problem is configuration drift and state management, and, as an example, the state file in Terraform contains deployed resources and managing this file poorly, or even hand-editing it, can cause an imbalanced infrastructure or may not deploy at all.     

"""

    test_single_predict(sample_text)

