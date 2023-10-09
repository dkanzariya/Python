MOD = 998244353

def sum_of_scores(n, a):
    max_a = [0] * n
    stack = []

    for i in range(n):
        while stack and a[i] > a[stack[-1]]:
            stack.pop()
        if stack:
            max_a[i] = max_a[stack[-1]] + 1
        else:
            max_a[i] = 0
        stack.append(i)

    ans = 0
    for i in range(n):
        ans += max(a[i], max_a[i]) % MOD

    return ans % MOD

# Read input
n = 4
a = [19, 14, 19, 9]

# Calculate and print the sum of scores
result = sum_of_scores(n, a)
print(result)
