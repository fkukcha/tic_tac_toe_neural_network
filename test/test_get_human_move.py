import unittest

from unittest.mock import patch
from io import StringIO
from api.get_human_move import get_human_move


class TestGetHumanMove(unittest.TestCase):

    @patch('builtins.input', side_effect=['1'])
    def test_valid_move(self, mock_input):
        """Test that a valid move is returned. """
        board = ['-', '-', '-', '-', '-', '-', '-', '-', '-']
        expected_move = 0
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            move = get_human_move(board)
            self.assertEqual(move, expected_move)
            self.assertEqual(mock_stdout.getvalue().strip(), '')  # No output expected on valid move

    @patch('builtins.input', side_effect=['a', '9', '3'])
    def test_invalid_input_then_valid(self, mock_input):
        """Test that an invalid input is handled correctly."""
        board = ['-', '-', '-', '-', '-', '-', '-', '-', '-']
        expected_move = 8
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            move = get_human_move(board)
            self.assertEqual(move, expected_move)
            self.assertIn("Invalid move", mock_stdout.getvalue().strip())

    @patch('builtins.input', side_effect=['10', '0', '4'])
    def test_out_of_range_input_then_valid(self, mock_input):
        """Test that an out-of-range input is handled correctly."""
        board = ['-', '-', '-', '-', '-', '-', '-', '-', '-']
        expected_move = 3
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            move = get_human_move(board)
            self.assertEqual(move, expected_move)
            self.assertIn("Invalid move", mock_stdout.getvalue().strip())


if __name__ == '__main__':
    unittest.main()
