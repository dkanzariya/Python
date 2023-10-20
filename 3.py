def string(list1):
    n = len(list1)
    final = 0
    for i in range(n):
        # print(list1[i])
        # print(10 ** (n-i-1) * list1[i])
        final += 10 ** (n-i-1) * list1[i]
        print(final)
    final = final + 1
    list_final = [int(j) for j in str(final)]
    return list_final

list = [1, 2, 3, 4]
print(string(list))