def maxScore(N, K, arr):
    max_score = 0
    current_max = -1
    current_partition = []
    max_list = []
    for i in range(K):
        for num in arr:

            current_max = max(current_max, num)
            current_partition.append(num)

            if len(current_partition) >= K+i:
                max_score += current_max
                current_partition = []
                current_max = -1
        max_list.append(max_score)
        max_score = 0

    return max(max_list)


# # Example 1
# N1 = 4
# K1 = 1
# arr1 = [1, 2, 3, 4]
# print(maxScore(N1, K1, arr1))  # Output: 10

# Example 2
N2 = 6
K2 = 2
arr2 = [1, 2, 9, 8, 3, 4]
print(maxScore(N2, K2, arr2))  # Output: 17
