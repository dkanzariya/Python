from typing import List

class Solution:
    def findIndices(self, nums: List[int], indexDifference: int, valueDifference: int) -> List[int]:
        n = len(nums)
        i = 0
        j = i + indexDifference

        while i <= n and j <= n:
            print(i, j)
            if i > n or j > n:
                j = n
                i = n
            if j == n:
                i += 1
                j = indexDifference
            if abs(nums[i] - nums[j]) >= valueDifference:
                return [i, j]

            j += 1
        return [-1, -1]


solution = Solution()
nums = [1,2,3]
indexDifference = 2
valueDifference = 4
print(solution.findIndices(nums, indexDifference,valueDifference))  # Output: "11001"