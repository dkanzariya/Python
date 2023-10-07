def max_triplet_value(nums):
  max_value = 0
  max1, min2 = 0, float('inf')

  for num in nums:

    if max1 != 0 and num >= max1:
      max_value = max(max_value, (max1 - min2) * num)
    elif num >= max1:
      max1 = num
    elif num <= min2:
      min2 = num


  return max_value

# Example usage:

nums = [1,10,3,4,19]
max_value = max_triplet_value(nums)
print(max_value)
