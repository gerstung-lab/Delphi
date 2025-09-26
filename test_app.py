import unittest
from fastapi.testclient import TestClient
from app import app

class TestDelphiAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_extrapolate_trajectory(self):
        # Prepare the example trajectory
        trajectory = [
            {"event": "Male", "age": 0},
            {"event": "B01 Varicella [chickenpox]", "age": 2},
            {"event": "L20 Atopic dermatitis", "age": 3},
            {"event": "No event", "age": 5},
            {"event": "No event", "age": 10},
            {"event": "No event", "age": 15},
            {"event": "No event", "age": 20},
            {"event": "G43 Migraine", "age": 20},
            {"event": "E73 Lactose intolerance", "age": 21},
            {"event": "B27 Infectious mononucleosis", "age": 22},
            {"event": "No event", "age": 25},
            {"event": "J11 Influenza, virus not identified", "age": 28},
            {"event": "No event", "age": 30},
            {"event": "No event", "age": 35},
            {"event": "No event", "age": 40},
            {"event": "Smoking low", "age": 41},
            {"event": "BMI mid", "age": 41},
            {"event": "Alcohol low", "age": 41},
            {"event": "No event", "age": 42}
        ]
        # POST request to the API
        response = self.client.post("/extrapolate_trajectory", json=trajectory)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("trajectory", data)
        self.assertIsInstance(data["trajectory"], list)
        # Check that the first event matches the input
        self.assertEqual(data["trajectory"][0]["event"], "Male")
        # Optionally, check the length of the output trajectory
        self.assertGreaterEqual(len(data["trajectory"]), len(trajectory))

if __name__ == "__main__":
    unittest.main()
