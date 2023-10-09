def find_array(P):
    if P < 41:
        return -1

    array = []
    remaining_sum = 41

    for i in range(9, 1, -1):
        while P % i == 0 and remaining_sum - i >= P // i - 1:
            array.extend([i] * (P // i))
            remaining_sum -= i * (P // i)
            P //= i

    if remaining_sum == 0:
        return array
    else:
        return -1

# Example usage:
P = 2023
array = find_array(P)

if array == -1:
    print("Case #1: -1")
else:
    print("Case #1: {} {}".format(len(array), " ".join([str(x) for x in array])))
