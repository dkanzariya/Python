def compute_lps(string):
    n = len(string)
    lps = [0] * n
    length = 0  # Length of the previous longest prefix suffix

    i = 1
    while i < n:
        if string[i] == string[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1

    return lps

def find_lps_for_all_substrings(s):
    n = len(s)
    lps_values = []

    for i in range(n):
        substring = s[:i + 1]
        lps = compute_lps(substring)
        lps_value = lps[-1] if lps else "none"  # Use "none" when no proper LPS exists
        lps_values.append(lps_value)

    return lps_values

# Read input string
s = input()
re = []
# Find and print LPS for all substrings
lps_values = find_lps_for_all_substrings(s)
for i, value in enumerate(lps_values):
	if value != 0:
		print("The LPS for {} is {}".format(s[:i+1], s[:value]))
	else :
		re.append("none")
	if len(re) == len(s):
		print("none")