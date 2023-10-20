def countPrimes(n):
    count = 0

    if n <= 2:
        return 0

    # if n < 3:
    # return 1

    isprime = [True] * n
    isprime[0] = isprime[1] = False

    for p in range(2, int(n ** 0.5) + 1):
        print(int(n ** 0.5) + 1)
        print(p)
        if isprime[p]:
            for i in range(p * p, n, p):
                isprime[i] = False
                # print(i, p)
    return sum(isprime)
print(countPrimes(10))
