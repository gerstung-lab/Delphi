import unittest
from fastapi.testclient import TestClient
from app import app

class TestDelphiAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_extrapolate_trajectory(self):
        # Prepare the example trajectory (ages in years, will be converted to days by multiplying by 365.25)
        trajectory = [
            {"event": "Male", "age": 0 * 365.25},
            {"event": "B01 Varicella [chickenpox]", "age": 2 * 365.25},
            {"event": "L20 Atopic dermatitis", "age": 3 * 365.25},
            {"event": "No event", "age": 5 * 365.25},
            {"event": "No event", "age": 10 * 365.25},
            {"event": "No event", "age": 15 * 365.25},
            {"event": "No event", "age": 20 * 365.25},
            {"event": "G43 Migraine", "age": 20 * 365.25},
            {"event": "E73 Lactose intolerance", "age": 21 * 365.25},
            {"event": "B27 Infectious mononucleosis", "age": 22 * 365.25},
            {"event": "No event", "age": 25 * 365.25},
            {"event": "J11 Influenza, virus not identified", "age": 28 * 365.25},
            {"event": "No event", "age": 30 * 365.25},
            {"event": "No event", "age": 35 * 365.25},
            {"event": "No event", "age": 40 * 365.25},
            {"event": "Smoking low", "age": 41 * 365.25},
            {"event": "BMI mid", "age": 41 * 365.25},
            {"event": "Alcohol low", "age": 41 * 365.25},
            {"event": "No event", "age": 42 * 365.25}
        ]
        # POST request to the API
        response = self.client.post("/extrapolate_trajectory", json=trajectory)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("trajectory", data)
        self.assertIsInstance(data["trajectory"], list)
        
        # Check that the input trajectory is included in the output
        for i, input_event in enumerate(trajectory):
            output_event = data["trajectory"][i]
            self.assertEqual(output_event["event"], input_event["event"])
            self.assertAlmostEqual(output_event["age"], input_event["age"] / 365.25, places=1)
        
        # Check that there are generated events after the input
        generated_events = data["trajectory"][len(trajectory):]
        self.assertGreater(len(generated_events), 0, "Should have generated additional events")
        
        # Check that generated events have reasonable ages (after age 42)
        for event in generated_events:
            self.assertGreater(event["age"], 42, "Generated events should be after the last input age")
            self.assertLessEqual(event["age"], 100, "Generated events should be within reasonable age range")
        
        # Check that the trajectory ends with "Death" as expected
        last_event = data["trajectory"][-1]
        self.assertEqual(last_event["event"], "Death", "Trajectory should end with Death")
        
        # Check for some expected disease patterns in generated trajectory
        generated_event_names = [e["event"] for e in generated_events]
        self.assertTrue(any("hypertension" in event.lower() for event in generated_event_names), 
                       "Should include hypertension-related events")
        self.assertTrue(any("no event" in event.lower() for event in generated_event_names), 
                       "Should include 'No event' entries")

if __name__ == "__main__":
    unittest.main()
