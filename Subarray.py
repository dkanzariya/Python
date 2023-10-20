def find_subarrays(arr):
    subarrays = []
    result_list = []
    n = len(arr)
    result = 0
    for start in range(n):
        for end in range(start, n):
            subarray = arr[start:end + 1]
            subarrays.append(subarray)
            element_count = {}
            for element in subarray:
                if element in element_count:
                    element_count[element] += 1
                    result += 1
                else:
                    element_count[element] = 1
        result_list.append(result)
    return result_list


# Example usage:
N = 10
arr = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]
result = find_subarrays(arr)
set_arr = set(arr)
length = len(set_arr)
print(result[length-1])
