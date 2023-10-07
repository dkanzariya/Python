
def maximumTripletValue(nums):
    max_value = 0

    # Iterate over all pairs of indices (i, j) such that i < j.
    for i in range(len(nums) - 2):
        for j in range(i + 1, len(nums) - 1):
            # Find the maximum value of (nums[i] - nums[j]) * nums[k] for all k such that k > j.
            max_k_value = 0
            max_k_value = max(max_k_value, (nums[i] - nums[j]) * nums[j + 1])

            # Update the maximum value.
            max_value = max(max_value, max_k_value)

    return max_value


nums = [12, 6, 1, 2, 7]
max_value = maximumTripletValue(nums)
print(max_value)