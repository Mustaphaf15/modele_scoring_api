import unittest
from fastapi.testclient import TestClient
from app import app


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_read_root(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Bienvenue", response.json()["message"])

    def test_list_id(self):
        response = self.client.get("/list_id")
        self.assertEqual(response.status_code, 200)
        self.assertIn("SK_ID_CURR", response.json())

    def test_predict_valid_client(self):
        # Replace 100001 with an actual SK_ID_CURR from your CSV
        response = self.client.get("/get_predict/100001")
        if response.status_code == 200:
            self.assertIn("SK_ID_CURR", response.json())
        elif response.status_code == 404:
            self.assertIn("non trouvé", response.json()["detail"])

    def test_predict_invalid_client(self):
        # Assuming this ID doesn't exist
        response = self.client.get("/get_predict/999999")
        self.assertEqual(response.status_code, 404)
        self.assertIn("non trouvé", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
