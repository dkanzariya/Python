class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        dict = {}
        for i in nums:
            if i in dict:
                dict[i] += 1
            else:
                dict[i] = 1
        elements_with_frequency_1 = [ele for ele, freq in dict.items() if freq == 1]
        if elements_with_frequency_1:
            return random.choice(elements_with_frequency_1)


class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        ans = list(set(nums))
        if len(nums) == len(ans):
            return False
        else:
            return True


class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        for i in range(k):
            ele = nums[-1]
            nums.pop(-1)
            nums.insert(0, ele)

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        final = 0
        for i in range(1, n):
            if prices[i - 1] < prices [i]:
                final += prices[i] - prices [i - 1]
        return final


class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if not nums:
            return 0  # If the list is empty, there are no unique elements.

        # Initialize a pointer for unique elements.
        unique_ptr = 0

        for i in range(1, len(nums)):
            if nums[i] != nums[unique_ptr]:
                unique_ptr += 1
                nums[unique_ptr] = nums[i]

        return unique_ptr + 1  # The length of unique elements.

        # nums = list(set(nums))
        # return nums
        '''
        l = len(nums)
        ans = []
        count = 1
        ans = list(set(nums))
        return len(ans)
        '''


class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        dict = {}
        for i in nums1:
            if i in dict:
                dict[i] += 1
            else:
                dict[i] = 1
        dict2 = {}
        for i in nums2:
            if i in dict2:
                dict2[i] += 1
            else:
                dict2[i] = 1

        res = []
        for i in nums2:
            if i in dict and dict[i] > 0:
                res.append(i)
                dict[i] -= 1

        # inter = set(list1) & set(list2)

        return res
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        n = len(digits)
        final = 0
        for i in range(n):
            # print(list1[i])
            # print(10 ** (n-i-1) * list1[i])
            final += 10 ** (n-i-1) * digits[i]
            # print(final)
        final = final + 1
        list_final = [int(j) for j in str(final)]
        return list_final


class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """

        new_list = [element for element in nums if element != 0]
        for i in range(len(new_list)):
            nums[i] = new_list[i]

        for j in range(len(nums) - len(new_list)):
            nums.pop(len(nums) - j - 1)
            nums.append(0)

        '''
        count = 0
        for i in nums:
            if i == 0:
                nums.pop(i)
                count += 1
        for i in range(count):
            nums.append(0)
        '''


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i in range(len(nums)):
            for j in range((i + 1), len(nums)):
                if (nums[i] + nums[j] == target):
                    return [i, j]

class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        def has_duplicate(arr):
            seen = set()
            for num in arr:
                if num != ".":
                    if num in seen:
                        return True
                    seen.add(num)
            return False

        for i in range(9):
            # Check rows
            if has_duplicate(board[i]):
                return False

            # Check columns
            if has_duplicate([board[j][i] for j in range(9)]):
                return False

        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                # Check 3x3 sub-grids
                subgrid = [board[x][y] for x in range(i, i + 3) for y in range(j, j + 3)]
                if has_duplicate(subgrid):
                    return False

        return True
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        left = 0
        right = len(s) - 1
        while left < right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1


class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        count = 0
        for i in range(len(haystack) - len(needle) + 1):
            print(haystack[i: i + len(needle)])
            if haystack[i: i + len(needle)] == needle:
                return i
        return -1


class Solution:
    def myAtoi(self, s: str) -> int:
        s = s.strip()
        if not s:
            return 0
        sign = 1
        if s[0] == '-' or s[0] == '+':
            sign = -1 if s[0] == '-' else 1
            s = s[1:]

        value = 0
        for c in s:
            if not c.isdigit():
                break
            print(c)
            value = value * 10 + (ord(c) - ord('0'))
        print(value)
        value = max(-2 ** 31, min(value * sign, 2 ** 31 - 1))

        return value


class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        strs.sort()

        res = ""

        for i in range(len(strs[0])):
            if strs[0][i] == strs[-1][i]:
                res += strs[0][i]
            else:
                break
        return res


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        if node.next is None:
            return

        node.val = node.next.val
        node.next = node.next.next


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode(0)

        dummy.next = head
        first = dummy
        second = dummy
        for i in range(0, n + 1):
            first = first.next

        while first is not None:
            first = first.next
            second = second.next

        second.next = second.next.next
        return dummy.next


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        current = head

        while current is not None:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        return prev


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()

        current = dummy

        while list1 and list2:
            if list1.val < list2.val:
                current.next = list1
                list1 = list1.next
            else:
                current.next = list2
                list2 = list2.next
            current = current.next

        if list1:
            current.next = list1
        if list2:
            current.next = list2
        return dummy.next


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:

        def reverselist(node):
            prev = None
            current = node

            while current:
                next_n = current.next
                current.next = prev
                prev = current
                current = next_n

            return prev

        if not head:
            return True

        slow = head
        fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        sec_half = reverselist(slow)

        first_half = head

        while sec_half:
            if first_half.val != sec_half.val:
                return False
            first_half = first_half.next
            sec_half = sec_half.next
        return True


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if not head or not head.next:
            return False

        slow = head
        fast = head

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if root is None:
            return 0

        leftd = self.maxDepth(root.left)
        rightd = self.maxDepth(root.right)

        return max(leftd, rightd) + 1

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def is_valid(node, lower=float('-inf'), upper=float('inf')):
            if not node:
                return True

            if not (lower < node.val < upper):
                return False

            return is_valid(node.left, lower, node.val) and is_valid(node.right, node.val, upper)

        return is_valid(root)


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return True

        def isMirror(left, right):
            if not left and not right:
                return True
            if not left or not right:
                return False
            return (left.val == right.val) and isMirror(left.left, right.right) and isMirror(left.right, right.left)

        return isMirror(root.left, root.right)


from collections import deque


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []

        result = []  # To store the final level order traversal
        queue = deque([root])  # Initialize a deque with the root node

        while queue:
            level_vals = []  # To store values at the current level
            level_size = len(queue)  # Determine the number of nodes at this level

            for i in range(level_size):
                node = queue.popleft()  # Dequeue the first node in the queue
                level_vals.append(node.val)  # Add its value to the current level

                # Enqueue left and right children if they exist
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            result.append(level_vals)  # Append the values at the current level

        return result
        '''
        if not root:
            return []

        result = []
        queue = deque([root])

        while queue:
            level_vals = []
            size = len(queue)

            for i in range(size):
                node = queue.popleft()
                level_vals.append(node.val)

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            result.append(level_vals)

        return result
        '''


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        if not nums:
            return None

        middle = len(nums) // 2

        root = TreeNode(nums[middle])

        root.left = self.sortedArrayToBST(nums[:middle])
        root.right = self.sortedArrayToBST(nums[middle + 1:])

        return root


# The isBadVersion API is already defined for you.
# def isBadVersion(version: int) -> bool:

class Solution:
    def firstBadVersion(self, n: int) -> int:

        left, right = 1, n

        while left < right:
            mid = left + (right - left) // 2

            if isBadVersion(mid):
                right = mid
            else:
                left = mid + 1
        return left

        '''
        for i in range(1, n+1):
            if isBadVersion(i):
                return i
        '''


class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 1:
            return 1
        if n == 2:
            return 2

        ways = [0] * n

        ways[0] = 1
        ways[1] = 2

        for i in range(2, n):
            ways[i] = ways[i - 1] + ways[i - 2]

            print(ways)
        return ways[-1]


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0
        final = 0
        min_price = prices[0]
        for price in prices:
            min_price = min(min_price, price)

            profit = price - min_price

            final = max(final, profit)
        return final


class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0

        current_sum = max_sum = nums[0]

        for num in nums[1:]:
            current_sum = max(num, current_sum + num)
            max_sum = max(max_sum, current_sum)

        return max_sum

        """
        final = float("-inf")

        if len(nums) == 1:
            return nums[0]

        for i in range(len(nums)):
            # sum1 = 0
            for j in range(i, len(nums) + 1):
                if nums[i:j] == []:
                    continue
                else:
                    sum1 = sum(nums[i:j])
                    final = max(sum1, final)
        return final
        """


class Solution:
    def rob(self, nums: List[int]) -> int:
        if not nums:
            return 0

        if len(nums) == 1:
            return nums[0]

        rob, skip = nums[0], 0

        for i in range(1, len(nums)):
            robn = skip + nums[i]
            skipn = max(rob, skip)

            rob, skip = robn, skipn
        return max(rob, skip)

        '''
        num1 = []
        num2 = []
        for i in range(len(nums)):
            if i % 2 == 0:
                num2.append(nums[i])
            else:
                num1.append(nums[i])
        return max(sum(num1), sum(num2))
        '''


import random


class Solution:

    def __init__(self, nums: List[int]):
        self.original = list(nums)
        self.shuffled = list(nums)

    def reset(self) -> List[int]:
        return self.original

    def shuffle(self) -> List[int]:
        n = len(self.shuffled)

        for i in range(n - 1, 0, -1):
            j = random.randint(0, i)
            self.shuffled[i], self.shuffled[j] = self.shuffled[j], self.shuffled[i]
        return self.shuffled

# Your Solution object will be instantiated and called as such:
# obj = Solution(nums)
# param_1 = obj.reset()
# param_2 = obj.shuffle()

class MinStack:

    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val: int) -> None:
        self.stack.append(val)

        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self) -> None:
        if self.stack:
            if self.stack[-1] == self.min_stack[-1]:
                self.min_stack.pop()
            self.stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]

# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
class Solution:
    def fizzBuzz(self, n: int) -> List[str]:
        answer = []
        for i in range(1, n + 1):
            if (i % 3 == 0  and i % 5 == 0):
                answer.append("FizzBuzz")
                #return "FizzBuzz"
            elif (i % 3 == 0):
                answer.append("Fizz")
                #return "Fizz"
            elif (i%5 == 0):
                answer.append("Buzz")
            else:
                answer.append(str(i))
        return answer


class Solution:
    def countPrimes(self, n: int) -> int:
        count = 0

        if n <= 2:
            return 0

        # if n < 3:
        # return 1

        isprime = [True] * n
        isprime[0] = isprime[1] = False

        for p in range(2, int(n ** 0.5) + 1):
            if isprime[p]:
                for i in range(p * p, n, p):
                    isprime[i] = False
        return sum(isprime)


import math


class Solution:
    def isPowerOfThree(self, n: int) -> bool:
        if n <= 0:
            return False

        return (1162261467) % n == 0

        '''
        log3n = math.log(n, 3)

        return int(log3n) == log3n


        while n % 3 == 0:
            n //= 3

        return n == 1
        '''


class Solution:
    def romanToInt(self, s: str) -> int:
        dict = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}

        total = 0
        prev = 0

        for char in s[::-1]:
            value = dict[char]
            if value < prev:
                total -= value
            else:
                total += value
            prev = value
        return total


class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0

        while n:
            count += n & 1
            n >>= 1
        return count
        '''
        n = bin(n)

        count = 0

        for bit in n[2:]:
            if bit == '1':
                count += 1
        return count
        '''

        '''
        n = str(n)
        count = 0
        for i in range(31):
            if n[i] == '1':
                count += 1
        return count
        '''


class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        xorres = x ^ y
        dis = 0

        while xorres:
            dis += xorres & 1
            xorres >>= 1
        return dis

        '''
        i = 0
        k = 0
        j = 0
        count  = 0
        while count == 0:
            count += x & 1
            x >>= 1
            i += 1
        while j == 0:
            j += y & 1
            y >>= 1
            k += 1
        print(i, k)

        res = abs(i-k)
        if res == 0:
            return 1

        return res
        '''


class Solution:
    def reverseBits(self, n: int) -> int:
        result = 0

        for i in range(32):
            result <<= 1
            result |= n & 1
            n >>= 1
        return result

        '''
        s = str(n)

        s = s[::-1]

        print(s)

        num = int(s, 2)

        return num
        '''


class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        trya = []

        for i in range(numRows):
            row = [1] * (i + 1)

            if i >= 1:
                for j in range(1, i):
                    row[j] = trya[i - 1][j - 1] + trya[i - 1][j]
            trya.append(row)

        return trya


class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        mapping = {')': '(', '}': '{', ']': '['}

        for char in s:
            if char in mapping:
                top_element = stack.pop() if stack else ''
                if top_element != mapping[char]:
                    return False
            else:
                stack.append(char)

        return not stack


'''
        for char in s:
            if char == '(':
                stack.append(char)
            if char == '{':
                stack.append(char)
            if char == '[':
                stack.append(char)
            # if char == '(':
                # stack.append(char)

            if char == ')':
                stack.pop(char)
            if char == ')':
                stack.pop(char)
            if char == ')':
                stack.pop(char)
        return stack == []
'''
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        for i in range(len(nums) + 1):
            if i in nums:
                continue
            else:
                return i
