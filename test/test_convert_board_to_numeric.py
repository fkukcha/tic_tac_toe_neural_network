from api.convert_board_to_numeric import convert_board_to_numeric


def test_convert_board_to_numeric():
    # Test case 1: Mixed board
    board = ['X', 'O', '-', '-', 'X', '-', '-', 'O', 'X']
    expected_output = [1, -1, 0, 0, 1, 0, 0, -1, 1]
    assert convert_board_to_numeric(board) == expected_output

    # Test case 2: All X
    board = ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X']
    expected_output = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert convert_board_to_numeric(board) == expected_output

    # Test case 3: All O
    board = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    expected_output = [-1, -1, -1, -1, -1, -1, -1, -1, -1]
    assert convert_board_to_numeric(board) == expected_output

    # Test case 4: Empty board
    board = ['-', '-', '-', '-', '-', '-', '-', '-', '-']
    expected_output = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert convert_board_to_numeric(board) == expected_output

    # Test case 5: Mixed board with different ordering
    board = ['O', 'X', 'X', 'O', '-', 'X', '-', 'O', '-']
    expected_output = [-1, 1, 1, -1, 0, 1, 0, -1, 0]
    assert convert_board_to_numeric(board) == expected_output
