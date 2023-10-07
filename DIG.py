def digital_root(x):
    while x >= 10:
        x = sum(map(int, str(x)))
    return x


def MaxVal(N, A, B):
    DIG = []
    for i in range(N):
        # print(A[i] ^ B[i])
        x = ((A[i] ** B[i]) + (B[i] ** A[i]))
        DIG.append(digital_root(x))

    # Sort DIG in non-decreasing order
    DIG.sort()

    # Concatenate sorted DIG values to form the maximum number
    result = int("".join(map(str, DIG[::-1])))

    return result


def main():
    N = 2
    A = [2, 2]
    B = [2, 2]
    # for j in range(N):
    #     A.append(int(input()))
    # for j in range(N):
    #     B.append(int(input()))

    result = MaxVal(N, A, B)
    print(result)


if __name__ == "__main__":
    main()
