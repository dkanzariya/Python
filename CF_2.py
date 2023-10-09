def minimum_announcement_cost(n, p, a, b):
    residents = [(a[i], b[i]) for i in range(n)]
    residents.sort(key=lambda x: x[1], reverse=True)

    min_cost = 0
    remaining_residents = n
    direct_announcement = 0

    for i in range(n):
        if direct_announcement < remaining_residents * p:
            min_cost += direct_announcement
            remaining_residents -= 1
            direct_announcement += p
        else:
            break

        while residents[i][0] > 0:
            if direct_announcement < remaining_residents * p:
                min_cost += residents[i][1]
                direct_announcement += p
                residents[i] = (residents[i][0] - 1, residents[i][1])
            else:
                break

    return min_cost


# Read the number of test cases
# t = int(input())
t = 1
# Iterate through each test case
for _ in range(t):
    # n, p = map(int, input().split())
    # a = list(map(int, input().split()))
    # b = list(map(int, input().split()))
    n, p = 6, 3
    a = [2, 3, 2, 1, 1, 3]
    b = [4, 3, 2, 6, 3, 6]
    # Calculate and print the minimum cost
    result = minimum_announcement_cost(n, p, a, b)
    print(result)
