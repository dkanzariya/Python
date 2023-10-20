def maxSubArray(nums):
    if len(nums) == 0:
        return 0

    current_sum = max_sum = nums[0]

    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
        print(current_sum, max_sum)
    return max_sum

nums = [-2,-1]
# print(sum(nums[1:2]))
print(maxSubArray(nums))