def lengthOfLongestSubstring(s):
    char_index = {}
    maxl = 0
    start = 0

    for end, char in enumerate(s):
        if char in char_index and char_index[char] >= start:
            start = char_index[char] + 1
        char_index[char] = end
        maxl = max(maxl, end-start+1)
    return maxl

s = "abcabcbb"
print(lengthOfLongestSubstring(s))
