import unittest
from unittest.mock import patch
from oldbatch.batchpredict import get_predictions_for_valid_cvm_codes

class TestBatchPredict(unittest.TestCase):
    @patch('batchpredict.get_predictions')
    @patch('batchpredict.check_data_availability')
    def test_get_predictions_for_valid_cvm_codes(self, mock_check_data_availability, mock_get_predictions):
        # Mock the return value of check_data_availability
        mock_check_data_availability.return_value = ['cvm_code1', 'cvm_code2']
        
        # Mock the return value of get_predictions
        mock_get_predictions.side_effect = [['prediction1'], ['prediction2']]
        
        # Call the function
        predictions = get_predictions_for_valid_cvm_codes('extended_company_data.csv')
        
        # Assert the expected results
        self.assertEqual(predictions, {'cvm_code1': ['prediction1'], 'cvm_code2': ['prediction2']})
        
        # Assert that check_data_availability was called with the correct argument
        mock_check_data_availability.assert_called_once_with('extended_company_data.csv')
        
        # Assert that get_predictions was called with the correct arguments
        mock_get_predictions.assert_any_call('cvm_code1')
        mock_get_predictions.assert_any_call('cvm_code2')

if __name__ == '__main__':
    unittest.main()