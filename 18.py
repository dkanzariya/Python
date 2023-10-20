def hammingDistance(x, y):
    i = 0
    k = 0
    j = 0
    count  = 0
    while count == 0:
        count += x & 1
        x >>= 1
        i += 1
    while j == 0:
        j += y & 1
        y >>= 1
        k += 1
    print(i, k)
    res = abs(i - k)
    if res == 0:
        return 1
    return abs(i - k)

print(hammingDistance(93, 73))
print(bin(93), bin(73))