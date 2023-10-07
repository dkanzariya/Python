# Function to find the smallest lexicographical string
def smallest_lexicographical_string(N, test_cases):
    results = []
    for i in range(N):
        K, S = test_cases[i]
        result = []
        while len(S) > 0:
            max_chars = min(K, len(S))
            candidates = S[:max_chars]
            min_char = min(candidates)
            result.append(min_char)

            # Update S to remove the processed characters
            S = S.replace(min_char, "", 1)

        results.append("".join(result))
    return results

# Read input
N = 1
test_cases = []

for _ in range(N):
    # K, S = input().split()
    K = int(1)
    test_cases.append((2, "abab"))

# Find and print the smallest lexicographical strings
results = smallest_lexicographical_string(N, test_cases)

for result in results:
    print(result)
