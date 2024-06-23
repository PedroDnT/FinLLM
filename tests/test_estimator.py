import unittest
from unittest.mock import patch
import pandas as pd
from openaicall import get_predictions

class TestGetPredictions(unittest.TestCase):
    @patch('openaicall.fetch_income_statement')
    @patch('openaicall.fetch_balance_sheet')
    @patch('openaicall.create_prompt')
    @patch('openaicall.openai.chat.completions.create')
    def test_get_predictions(self, mock_create, mock_create_prompt, mock_fetch_balance_sheet, mock_fetch_income_statement):
        # Mock the necessary functions and data
        mock_fetch_income_statement.return_value = pd.DataFrame({
            'index': ['A', 'B', 'C'],
            '2020': [100, 200, 300],
            '2021': [150, 250, 350],
            '2022': [200, 300, 400]
        })
        mock_fetch_balance_sheet.return_value = pd.DataFrame({
            'index': ['A', 'B', 'C'],
            '2020': [1000, 2000, 3000],
            '2021': [1500, 2500, 3500],
            '2022': [2000, 3000, 4000]
        })
        mock_create_prompt.return_value = "Mock prompt"
        mock_create.return_value.choices[0].message.content.strip.return_value = "Mock rationale"
        
        # Call the function
        predictions = get_predictions("ABC", output_type="B")
        
        # Assert the expected output
        expected_predictions = [
            ('2022', 'Increase; Earnings Direction; Magnitude; Confidence Score'),
            ('2023', 'Increase; Earnings Direction; Magnitude; Confidence Score'),
            ('2024', 'Increase; Earnings Direction; Magnitude; Confidence Score')
        ]
        self.assertEqual(predictions, expected_predictions)

if __name__ == '__main__':
    unittest.main()