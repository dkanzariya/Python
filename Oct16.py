def removeDuplicates(nums) -> int:
    ans = list(set(nums))
    return ans

print(removeDuplicates([1, 1, 2]))