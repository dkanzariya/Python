import numpy as np
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

def generate_submatrices(matrix, target_sum):
    submatrices = []
    N = len(matrix)
    count = 0
    #sum = 0
    for i in range(N):
        for j in range(N):
            for x in range(i, N):

                for y in range(j, N):
                    submatrix = []
                    sum = 0
                    for p in range(i, x + 1):
                        row = []
                        for q in range(j, y + 1):
                            row.append(matrix[p][q])
                            sum += matrix[p][q]
                        submatrix.append(row)
                        if sum == target_sum:
                            count = 0
                    # submatrices.append(submatrix)
                    # sub_num = np.array(submatrices)


    return count

def countSubmatricesWithSum(matrix, target_sum):
    rows = len(matrix)
    cols = len(matrix[0])
    count = 0

    for i in range(rows):
        for j in range(cols):
            for x in range(i, rows):
                for y in range(j, cols):
                    submatrix_sum = 0
                    for p in range(i, x + 1):
                        for q in range(j, y + 1):
                            submatrix_sum += matrix[p][q]
                    if submatrix_sum == target_sum:
                        count += 1

    return count

# Input
N = 3
F = 2

matrix = [[5,   -1,   6], [-2,   3,   8], [7, 4, -9]]
# print("Enter the elements of the matrix:")
# for _ in range(N):
#     row = list(map(int, input().split()))
#     matrix.append(row)

target_sum = factorial(F)

result = countSubmatricesWithSum(matrix, target_sum)
print("Number of submatrices with sum equal to {}!: {}".format(F, result))
