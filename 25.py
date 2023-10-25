def groupAnagrams(strs):
    anagrams = {}

    for word in strs:
        key = "".join(sorted(word))

        if key not in anagrams:
            anagrams[key] = []

        anagrams[key].append(word)

    return list(anagrams.values())

strs1 = ["eat","tea","tan","ate","nat","bat"]
strs2 = [""]
strs3 = ["a"]

print(groupAnagrams(strs1))
print(groupAnagrams(strs2))
print(groupAnagrams(strs3))
