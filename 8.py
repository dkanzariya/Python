s = ["h","e","l","l","o"]
print(s)
s = [s[-i-1] for i in range(len(s))]
print(s)
left = 0
right = len(s) - 1
while left < right:
    print(left, right)
    s[left], s[right] = s[right], s[left]
    left += 1
    right -= 1
    print(left, right)
print(s)