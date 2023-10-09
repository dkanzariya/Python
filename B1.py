def find_array(P):
    result = []
    current = 100

    while P > 0 and len(result) < 100:
        if P % current == 0 and sum(result) + current <= 41:
            result.append(current)
            P //= current
        else:
            current -= 1

    if P == 1 and sum(result) == 41:
        return result
    else:
        return [-1]


def main():
    T = int(input().strip())
    for i in range(1, T + 1):
        P = int(input().strip())
        result = find_array(P)
        if result == [-1]:
            print(f"Case #{i}: -1")
        else:
            N = len(result)
            print(f"Case #{i}: {N} {' '.join(map(str, result))}")


if __name__ == "__main__":
    main()
