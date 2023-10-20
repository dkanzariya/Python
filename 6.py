def is_valid_sudoku(board):
    def has_duplicate(arr):
        seen = set()
        for num in arr:
            if num != ".":
                if num in seen:
                    return True
                seen.add(num)
        return False

    for i in range(9):
        # Check rows
        if has_duplicate(board[i]):
            return False

        # Check columns
        if has_duplicate([board[j][i] for j in range(9)]):
            return False

    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            # Check 3x3 sub-grids
            subgrid = [board[x][y] for x in range(i, i + 3) for y in range(j, j + 3)]
            # print(subgrid)
            if has_duplicate(subgrid):
                return False

    return True
valid_board = [
    ["5", "3", ".", ".", "7", ".", ".", ".", "."],
    ["6", ".", ".", "1", "9", "5", ".", ".", "."],
    [".", "9", "8", ".", ".", ".", ".", "6", "."],
    ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
    ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
    ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
    [".", "6", ".", ".", ".", ".", "2", "8", "."],
    [".", ".", ".", "4", "1", "9", ".", ".", "5"],
    [".", ".", ".", ".", "8", ".", ".", "7", "9"]
]

invalid_board = [
    ["8", "3", ".", ".", "7", ".", ".", ".", "."],
    ["6", ".", ".", "1", "9", "5", ".", ".", "."],
    [".", "9", "8", ".", ".", ".", ".", "6", "."],
    ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
    ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
    ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
    [".", "6", ".", ".", ".", ".", "2", "8", "."],
    [".", ".", ".", "4", "1", "9", ".", ".", "5"],
    [".", ".", ".", ".", "8", ".", ".", "7", "9"]
]

print(is_valid_sudoku(valid_board))  # Output: True
print(is_valid_sudoku(invalid_board))  # Output: False
