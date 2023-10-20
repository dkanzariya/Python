def find_smallest_dominant_substring(N, S):
    smallest_dom_substring = ""
    smallest_dom_substring_len = float('inf')

    for start in range(N - 1):
        freq_counter = {}
        for end in range(start, N):
            char = S[end]
            freq_counter[char] = freq_counter.get(char, 0) + 1
            length = end - start + 1

            # Check if the current substring satisfies the entropy criteria
            if length >= 2:
                max_freq = max(freq_counter.values())
                if max_freq > length // 2:
                    if length < smallest_dom_substring_len:
                        smallest_dom_substring = S[start:end + 1]
                        smallest_dom_substring_len = length

    if smallest_dom_substring:
        return smallest_dom_substring
    else:
        return "ZERO"


# Input
N = 5
S = "ccdde"


N = 5
S = "abcbe"

# N = 4
# S = "zerr"

# Find and print the smallest dominant substring
result = find_smallest_dominant_substring(N, S)
print(result)
