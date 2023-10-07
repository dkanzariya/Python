# Function to find the smallest lexicographical string
def find_smallest_lexicographical_string(K, S):
    S = list(S)  # Convert the string to a list of characters for easy manipulation
    n = len(S)
    min_str = S[:]

    for j in range(K):
        for i in range(1, n):
            if min_str >= S[:]:
                max_char = max(S[:K])
                idx = S.index(max_char)
                S = S[:idx] + S[idx + 1:] + [max_char]
                min_str = min(min_str, S[:])  # Update the minimum string
    return ''.join(min_str)  # Convert the list back to a string


# Input the number of test cases
N = 1
results = []

# Process each test case
for _ in range(N):
    # K, S = input().split()
    # K = int(K)
    result = find_smallest_lexicographical_string(2, "abab")
    results.append(result)

# Print the results
for result in results:
    print(result)
