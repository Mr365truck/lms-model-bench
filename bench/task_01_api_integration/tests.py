import pytest
import os
import sys
import urllib.parse

sys.path.insert(0, os.path.dirname(__file__))

def test_api_client():
    from solution import APIClient, APIError
    
    class MockResponse:
        def __init__(self, json_data, status_code, headers=None):
            self._json_data = json_data
            self.status_code = status_code
            self.headers = headers or {}

        def json(self):
            return self._json_data

        def raise_for_status(self):
            if self.status_code != 200:
                from requests import HTTPError
                raise HTTPError(f"Error {self.status_code}")

    calls = []
    def mock_get(url, headers=None, timeout=None, params=None):
        # Extract page number
        page_val = None
        if params and "page" in params:
            page_val = int(params["page"])
        else:
            parsed = urllib.parse.urlparse(url)
            query = urllib.parse.parse_qs(parsed.query)
            if "page" in query:
                page_val = int(query["page"][0])
                
        calls.append({"url": url, "headers": headers, "page": page_val})
        
        if page_val == 1:
            return MockResponse({"records": [{"id": 1, "name": "A"}], "has_more": True}, 200)
        elif page_val == 2:
            # Simulate a 429 rate limit on page 2 first attempt
            page_2_attempts = len([c for c in calls if c["page"] == 2])
            if page_2_attempts == 1:
                return MockResponse({}, 429, {"Retry-After": "1"})
            # Second attempt succeeds
            return MockResponse({"records": [{"id": 2, "name": "B"}], "has_more": False}, 200)
        else:
            return MockResponse({}, 404)

    import requests
    original_get = requests.get
    requests.get = mock_get

    try:
        client = APIClient("http://api.test", "my_token", max_retries=3)
        records = client.fetch_all_records("data")
        
        # Verification
        assert len(records) == 2
        assert records[0]["id"] == 1
        assert records[1]["id"] == 2
        
        # Verify auth header
        for call in calls:
            assert call["headers"]["Authorization"] == "Bearer my_token"
            
        # Verify 429 was handled and retried (should be 2 attempts for page 2)
        page_2_calls = [c for c in calls if c["page"] == 2]
        assert len(page_2_calls) == 2
        
    finally:
        requests.get = original_get

def test_api_client_error():
    from solution import APIClient, APIError
    
    class MockResponse:
        def __init__(self, status_code):
            self.status_code = status_code
        def json(self): return {}

    def mock_get(url, headers=None, timeout=None, params=None):
        return MockResponse(500)

    import requests
    original_get = requests.get
    requests.get = mock_get

    try:
        client = APIClient("http://api.test", "token")
        with pytest.raises(APIError) as exc_info:
            client.fetch_page("data", 1)
        assert exc_info.value.status_code == 500
    finally:
        requests.get = original_get
