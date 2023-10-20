def string(x):
    list1 = list(str(x))
    n = len(list1)
    final = 0
    for i in range(n):
        # print(list1[i])
        # print(10 ** (n-i-1) * list1[i])
        final += 10 ** (i) * (int(list1[i]))

    return final


x = 123
list = list(str(x))
print(list)
print(string(x))
