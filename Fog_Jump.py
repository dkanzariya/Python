def minJumpsToReachEnd(n, A):
    if n <= 1:
        return 0

    # Initialize an array to keep track of minimum jumps required to reach each index
    jumps = [float('inf')] * n

    # The minimum jumps required to reach the starting position is 0
    jumps[0] = 0

    for i in range(1, n):
        for j in range(i):
            # Check if it's possible to reach index i from index j
            if j + A[j] >= i:
                # Update the minimum jumps required at index i
                jumps[i] = min(jumps[i], jumps[j] + 1)

    return jumps[n - 1]

# Input reading and function call
# n = 3
# A = [2, 1, 1]

n = 5
A = [2, 3, 1, 1, 4]

# n = 5
# A = [1, 1, 1, 2, 4]

result = minJumpsToReachEnd(n, A)
print(result)
