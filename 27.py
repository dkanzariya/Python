def longPal(s):
    n = len(s)
    if n <= 1:
        return s
    start, maxl = 0, 1

    is_pal = [[False] * n for _ in range(n)]

    for i in range(n):
        is_pal[i][i] = True

    for i in range(n-1):
        if s[i] == s[i+1]:
            is_pal[i][i+1] = True
            start = i
            maxl = 2
    for length in range(3, n+1):
        for i in range(n - length + 1):
            j = i + length - 1

            if s[i] == s[j] and is_pal[i+1][j-1]:
                is_pal[i][j] = True
                if length > maxl:
                    start = 1
                    maxl = length
    return s[start:start+maxl]

print(longPal("ccc"))