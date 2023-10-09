def minimum_time_to_execute_all_tasks(processorTime, tasks):
    processorTime.sort()
    tasks.sort(reverse = True)
    new_list = [x for x in processorTime for _ in range(4)]
    result = []
    for i in range(len(new_list)):
        result.append(new_list[i] + tasks[i])

    return max(result)


# Example usage:
# processorTime1 = [8, 10]
# tasks1 = [2, 2, 3, 1, 8, 7, 4, 5]
# print(minimum_time_to_execute_all_tasks(processorTime1, tasks1))  # Output: 16

processorTime2 = [143,228,349,231,392]
tasks2 = [102,365,363,211,38,96,98,79,365,289,252,201,259,346,21,68,128,56,167,183]
print(minimum_time_to_execute_all_tasks(processorTime2, tasks2))  # Output: 23



