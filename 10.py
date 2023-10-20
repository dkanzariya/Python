def longest_common_prefix(strings):

  if not strings:
    return ""

  strings.sort()

  longest_common_prefix = ""
  for i in range(len(strings[0])):
    if strings[0][i] == strings[-1][i]:
      longest_common_prefix += strings[0][i]
    else:
      break

  return longest_common_prefix

strings = ["flower", "flow", "flight"]

longest_common_prefix = longest_common_prefix(strings)

print(longest_common_prefix)
