def maxIncreaseKeepingSkyline(grid):
    n = len(grid)
    max_row_heights = [max(row) for row in grid]
    max_col_heights = [max(grid[i][j] for i in range(n)) for j in range(n)]

    total_increase = 0
    for i in range(n):
        for j in range(n):
            total_increase += min(max_row_heights[i], max_col_heights[j]) - grid[i][j]

    return total_increase


# Input the number of rows and columns
n = 4

# Input the elements of the grid as a comma-separated string and convert to a 2D list
# grid_str = input()
grid = [[3, 0, 8, 4], [2, 4, 5, 7], [9, 2, 6, 3], [0, 3, 1, 0]]

# Calculate the maximum total sum of height increases
result = maxIncreaseKeepingSkyline(grid)

# Print the result
print(result)
