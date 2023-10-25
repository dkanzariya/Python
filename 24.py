def setZeroes(matrix):
    rows, columns = len(matrix), len(matrix[0])
    zeror, zeroc = set(), set()

    for i in range(rows):
        for j in range(columns):
            if matrix[i][j] == 0:
                zeror.add(i)
                zeroc.add(j)
    print(zeroc, zeror)
    for row in zeror:
        for j in range(columns):
            matrix[row][j] = 0

    for col in zeroc:
        for i in range(rows):
            matrix[i][col] = 0
    return matrix



# Example usage
matrix1 = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]

print(setZeroes(matrix1))