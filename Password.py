def count_passwords(m, n):
    MOD = 10**9 + 7

    # Initialize the total count
    total_count = 0

    # Iterate through each possible length in the range [m, n]
    for length in range(m, n + 1):
        # Count the number of passwords for the current length
        if length == m:
            # The first character must be a lowercase letter, and the last character must be a digit
            # For the characters in between, there are 36 choices (26 lowercase letters + 10 digits)
            count = 26 * 36**(length - 2) * 10
        else:
            # For lengths greater than m, the first character can also be a digit
            # In this case, there are 37 choices (26 lowercase letters + 10 digits) for the first character
            count = 26 * 36**(length - 2) * 10

        # Add the count to the total
        total_count = (total_count + count) % MOD

    return total_count

# Input minimum and maximum password lengths
m = 2
n = 3

# Calculate the number of passwords within the specified range
result = count_passwords(m, n)

# Print the result modulo 10^9 + 7
print(result)
