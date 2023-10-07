MOD = 10**9 + 7

def count_ways_to_accommodate_guests(n, k):
    # Initialize a 2D DP table
    dp = [[0] * (n + 1) for _ in range(n + 1)]

    # Base case: There's only one way to accommodate 0 guests in 0 rooms
    dp[0][0] = 1

    # Fill the DP table using bottom-up approach
    for guests in range(1, n + 1):
        for rooms in range(1, n + 1):
            # If there are more rooms than guests, we can't accommodate them all
            if rooms > guests:
                dp[guests][rooms] = 0
            else:
                # Two possibilities: accommodate a full room or leave it empty
                dp[guests][rooms] = (dp[guests - 1][rooms - 1] + dp[guests - 1][rooms] * rooms) % MOD

    # The result is in dp[n][k] because we want to accommodate all guests in k rooms
    return dp[n][k]

# Read input
n = 4
k = 2

# Calculate and print the number of ways modulo 10^9 + 7
result = count_ways_to_accommodate_guests(n, k)
print(result)
