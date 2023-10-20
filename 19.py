def reverseBits(n):
    s = str(n)

    s = s[::-1]

    print(s)

    num = int(s, 2)

    return num
print(reverseBits('11111111111111111111111111111101'))
