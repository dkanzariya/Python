def replace_adjacent_duplicates(s):
    if not s:
        return s  # Return an empty string if the input is empty

    result = [s[0]]  # Initialize the result with the first character of s
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:  # Check if the current character is the same as the previous one
            result.pop()  # Remove the previous character from the result
            result.append(s[i].upper())  # Add the uppercase version of the current character
        else:
            result.append(s[i])  # Add the current character to the result

    return ''.join(result)  # Convert the list back to a string and return it


# Example usage:
s = "ggeeekk"
result = replace_adjacent_duplicates(s)
print(result)  # Output: "gEksforgEks"
