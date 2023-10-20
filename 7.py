def rotate(matrix):
    """
    Do not return anything, modify matrix in-place instead.
    """
    n = len(matrix)

    for i in range(n):
        for j in range((i+1), n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
            print(i, j)
    print(matrix)
    for i in range(n):
        matrix[i].reverse()
    print(matrix)

matrix = [[1,2,3],[4,5,6],[7,8,9]]
rotate(matrix)
print(matrix)