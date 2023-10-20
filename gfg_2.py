def subarrayValue(N, arr):
    MOD = 10 ** 9 + 7
    result = 0

    for i in range(N):
        freq = {}
        value = 0

        for j in range(i, N):
            if arr[j] in freq:
                freq[arr[j]] += 1
                if freq[arr[j]] == 2:
                    value += 1
            else:
                freq[arr[j]] = 1

            if value > 0:
                result = (result + 1) % MOD

    return result


# Example 2
N2 = 4
arr2 = [1, 2, 1, 2]
print(subarrayValue(N2, arr2))  # Output: 4
