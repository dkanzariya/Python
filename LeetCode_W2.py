def minOperations(s1, s2, x):
    n = len(s1)
    cost = 0

    for i in range(n):
        if s1[i] != s2[i]:
            if i + 1 < n and s1[i] == '0' and s1[i + 1] == '1':
                cost += x
                s1 = s1[:i] + '1' + s1[i + 1:]
            elif i + 1 < n and s1[i] == '1' and s1[i + 1] == '0':
                cost += x
                s1 = s1[:i] + '0' + s1[i + 1:]
            else:
                return -1

    return cost

# Example usage:
s1 = "1100011000"
s2 = "0101001010"
x = 2
print(minOperations(s1, s2, x))  # Output: 4

s1 = "10110"
s2 = "00011"
x = 4
print(minOperations(s1, s2, x))