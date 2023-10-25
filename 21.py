class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        result = []
        nums.sort()

        for i in range(len(nums) - 2):
            # print(i)
            if i > 0 and nums[i] == nums[i - 1]:
                continue

            left, right = i + 1, len(nums) - 1
            # print(left, right)
            while left < right:
                total = nums[i] + nums[left] + nums[right]
                if total < 0:
                    left += 1
                elif total > 0:
                    right -= 1
                else:
                    result.append([nums[i], nums[left], nums[right]])
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    left += 1
                    right -= 1
        return result
        '''
        result = []
        nums.sort()

        for i in range(len(nums)-2):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            left, right = i+1, len(nums)-1
            while left < right:
                total = nums[i] + nums[left] + nums[right]
                if total < 0 :
                    left += 1
                elif total > 0 :
                    right -= 1
                else:
                    result.append([nums[i], nums[left], nums[right]])
                    while left < right and nums[i] == nums[left+1]:
                        left += 1
                    while left < right and nums[i] == nums[right-1]:
                        right -= 1
                    left += 1
                    right -= 1
        return result
        '''


class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        rows, columns = len(matrix), len(matrix[0])
        zeror, zeroc = set(), set()

        for i in range(rows):
            for j in range(columns):
                if matrix[i][j] == 0:
                    zeror.add(i)
                    zeroc.add(j)

        print(zeroc, zeror)
        for row in zeror:
            for j in range(columns):
                matrix[row][j] = 0

        for col in zeroc:
            for i in range(rows):
                matrix[i][col] = 0
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        anagrams = {}

        for word in strs:
            key = "".join(sorted(word))

            if key not in anagrams:
                anagrams[key] = []

            anagrams[key].append(word)

        return list(anagrams.values())


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        char_index = {}
        maxl = 0
        start = 0

        for end, char in enumerate(s):
            if char in char_index and char_index[char] >= start:
                start = char_index[char] + 1
            char_index[char] = end
            maxl = max(maxl, end - start + 1)
        return maxl

        '''
        dict = {}
        if len(s) == 1:
            return 1
        for char in s[1:]:

            if char not in dict:
                dict[char] = 0
            else:
                dict[char] += 1
        return len(list(dict.values()))
        '''

def lengthOfLongestSubstring(s):
    char_index = {}
    maxl = 0
    start = 0

    for end, char in enumerate(s):
        if char in char_index and char_index[char] >= start:
            start = char_index[char] + 1
        char_index[char] = end
        maxl = max(maxl, end-start+1)
    return maxl

s = "abcabcbb"
print(lengthOfLongestSubstring(s))


class Solution:
    def longestPalindrome(self, s: str) -> str:

        if not s:
            return ""

        n = len(s)
        start = 0
        max_length = 1

        # Initialize a 2D DP table to track palindromes
        dp = [[False] * n for _ in range(n)]

        # All substrings of length 1 are palindromes
        for i in range(n):
            dp[i][i] = True

        # Check for palindromes of length 2
        for i in range(n - 1):
            if s[i] == s[i + 1]:
                dp[i][i + 1] = True
                start = i
                max_length = 2

        # Check for palindromes of length greater than 2
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1  # Ending index of the current substring

                # Check if the current substring is a palindrome
                if dp[i + 1][j - 1] and s[i] == s[j]:
                    dp[i][j] = True
                    if length > max_length:
                        start = i
                        max_length = length

        return s[start:start + max_length]

        '''
        n = len(s)
        if n <= 1:
            return s
        start, maxl = 0, 1

        is_pal = [[False] * n for _ in range(n)]

        for i in range(n):
            is_pal[i][i] = True

        for i in range(n-1):
            if s[i] == s[i+1]:
                is_pal[i][i+1] = True
                start = i
                maxl = 2

        for length in range(3, n+1):
            for i in range(n - length + 1):
                j = i + length - 1

                if s[i] == s[j] and is_pal[i+1][j-1]:
                    is_pal[i][j] = True
                    if length > maxl:
                        start = 1
                        maxl = length
        if maxl == 1:
            return s
        return s[start:start+maxl]
        '''


class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        first = second = float('inf')

        for num in nums:
            if num <= first:
                first = num
            elif num <= second:
                second = num
            else:
                return True
        return False
class Solution:
    def countAndSay(self, n: int) -> str:
        if n == 1:
            return "1"

        prev = "1"

        for _ in range(n - 1):
            curr = ""
            i = 0

            while i < len(prev):
                count = 1
                while i < len(prev) - 1 and prev[i] == prev[i + 1]:
                    count += 1
                    i += 1
                curr += str(count) + prev[i]
                i += 1

            prev = curr

        return prev
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy_head = ListNode(0)
        current = dummy_head
        carry = 0

        while l1 or l2:
            x = l1.val if l1 else 0
            y = l2.val if l2 else 0
            total = x + y + carry

            carry = total // 10
            current.next = ListNode(total % 10)

            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
            current = current.next

        if carry > 0:
            current.next = ListNode(carry)

        return dummy_head.next
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head

        odd_head = head
        even_head = head.next
        odd_ptr = odd_head
        even_ptr = even_head

        while even_ptr and even_ptr.next:
            odd_ptr.next = even_ptr.next
            odd_ptr = odd_ptr.next
            even_ptr.next = odd_ptr.next
            even_ptr = even_ptr.next

        odd_ptr.next = even_head

        return odd_head
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        if not headA or not headB:
            return None

        # Helper function to get the length of a linked list
        def get_length(node):
            length = 0
            while node:
                length += 1
                node = node.next
            return length

        # Get the lengths of both linked lists
        lenA = get_length(headA)
        lenB = get_length(headB)

        # Reset the pointers to the beginning of the lists
        currA, currB = headA, headB

        # Adjust the longer list by moving its pointer forward
        if lenA > lenB:
            for _ in range(lenA - lenB):
                currA = currA.next
        elif lenB > lenA:
            for _ in range(lenB - lenA):
                currB = currB.next

        # Move both pointers until they intersect or reach the end
        while currA != currB:
            currA = currA.next
            currB = currB.next

        # Return the intersected node, or None if there's no intersection
        return currA
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        result = []
        stack = []
        current = root

        while current or stack:
            while current:
                stack.append(current)
                current = current.left
            current = stack.pop()
            result.append(current.val)
            current = current.right

        return result
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []

        result = []
        queue = [root]
        reverse = False

        while queue:
            level_vals = []
            next_level = []

            for node in queue:
                level_vals.append(node.val)
                if node.left:
                    next_level.append(node.left)
                if node.right:
                    next_level.append(node.right)

            if reverse:
                level_vals = level_vals[::-1]

            result.append(level_vals)
            reverse = not reverse
            queue = next_level

        return result
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if not preorder or not inorder:
            return None

        root_val = preorder.pop(0)
        root = TreeNode(root_val)
        root_index = inorder.index(root_val)

        root.left = self.buildTree(preorder, inorder[:root_index])
        root.right = self.buildTree(preorder, inorder[root_index + 1:])

        return root
"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

class Solution:
    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        if not root:
            return root

        queue = [root]

        while queue:
            level_size = len(queue)

            for i in range(level_size):
                node = queue.pop(0)

                if i < level_size - 1:
                    node.next = queue[0]

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

        return root
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        def inorder(node):
            nonlocal k
            if not node:
                return None

            left_result = inorder(node.left)
            if left_result is not None:
                return left_result

            k -= 1
            if k == 0:
                return node.val

            return inorder(node.right)

        return inorder(root)
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid:
            return 0

        def dfs(row, col):
            if row < 0 or row >= len(grid) or col < 0 or col >= len(grid[0]) or grid[row][col] == '0':
                return
            grid[row][col] = '0'  # Mark the current land as visited
            # Recursively explore the neighboring lands
            dfs(row + 1, col)
            dfs(row - 1, col)
            dfs(row, col + 1)
            dfs(row, col - 1)

        num_islands = 0
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if grid[row][col] == '1':
                    num_islands += 1
                    dfs(row, col)

        return num_islands
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []

        # Create a dictionary mapping each digit to its corresponding letters
        phone = {
            '2': 'abc',
            '3': 'def',
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv',
            '9': 'wxyz'
        }

        def backtrack(index, path):
            if index == len(digits):
                combinations.append("".join(path))
                return
            digit = digits[index]
            for letter in phone[digit]:
                path.append(letter)
                backtrack(index + 1, path)
                path.pop()

        combinations = []
        backtrack(0, [])
        return combinations


class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        def backtrack(s, left, right):
            if len(s) == 2 * n:
                combinations.append(s)
                return
            if left < n:
                backtrack(s + '(', left + 1, right)
            if right < left:
                backtrack(s + ')', left, right + 1)

        combinations = []
        backtrack('', 0, 0)
        return combinations
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def backtrack(first):
            if first == n:
                permutations.append(nums[:])
            for i in range(first, n):
                nums[first], nums[i] = nums[i], nums[first]
                backtrack(first + 1)
                nums[first], nums[i] = nums[i], nums[first]

        n = len(nums)
        permutations = []
        backtrack(0)
        return permutations
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        def backtrack(start, current_subset):
            subsets.append(current_subset[:])
            for i in range(start, len(nums)):
                current_subset.append(nums[i])
                backtrack(i + 1, current_subset)
                current_subset.pop()

        subsets = []
        backtrack(0, [])
        return subsets


class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        def dfs(row, col, word_index):
            if word_index == len(word):
                return True
            if row < 0 or row >= len(board) or col < 0 or col >= len(board[0]) or board[row][col] != word[word_index]:
                return False

            temp, board[row][col] = board[row][col], "/"
            found = (dfs(row + 1, col, word_index + 1) or
                     dfs(row - 1, col, word_index + 1) or
                     dfs(row, col + 1, word_index + 1) or
                     dfs(row, col - 1, word_index + 1))

            board[row][col] = temp
            return found

        for row in range(len(board)):
            for col in range(len(board[0])):
                if board[row][col] == word[0] and dfs(row, col, 0):
                    return True

        return False


class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        def dfs(row, col, word_index):
            if word_index == len(word):
                return True

            if (
                    row < 0 or row >= len(board) or
                    col < 0 or col >= len(board[0]) or
                    board[row][col] != word[word_index]
            ):
                return False

            temp, board[row][col] = board[row][col], "/"
            found = (
                    dfs(row + 1, col, word_index + 1) or
                    dfs(row - 1, col, word_index + 1) or
                    dfs(row, col + 1, word_index + 1) or
                    dfs(row, col - 1, word_index + 1)
            )
            board[row][col] = temp

            return found

        for row in range(len(board)):
            for col in range(len(board[0])):
                if board[row][col] == word[0] and dfs(row, col, 0):
                    return True

        return False


class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # nums.sort()
        red, white, blue = 0, 0, len(nums) - 1

        while white <= blue:
            if nums[white] == 0:
                # If the current element is 0, swap it with the element at the red pointer
                nums[red], nums[white] = nums[white], nums[red]
                red += 1
                white += 1
            elif nums[white] == 1:
                # If the current element is 1, move to the next element
                white += 1
            else:
                # If the current element is 2, swap it with the element at the blue pointer
                nums[white], nums[blue] = nums[blue], nums[white]
                blue -= 1
from collections import Counter
import heapq

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # Use Counter to count the frequency of each element
        count = Counter(nums)

        # Create a min-heap (priority queue) to keep track of k most frequent elements
        heap = []

        # Iterate through the unique elements in the Counter
        for num, freq in count.items():
            # Push the element into the heap
            heapq.heappush(heap, (freq, num))

            # If the size of the heap exceeds k, pop the smallest element
            if len(heap) > k:
                heapq.heappop(heap)

        # Extract the elements from the heap
        result = [elem[1] for elem in heap]

        # The result contains the k most frequent elements in ascending order
        return result


import heapq


class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        min_heap = []

        for num in nums:
            if len(min_heap) < k:
                heapq.heappush(min_heap, num)
            else:
                if num > min_heap[0]:
                    heapq.heappop(min_heap)
                    heapq.heappush(min_heap, num)

        return min_heap[0]
        '''
        def partition(left, right, pivot_index):
            pivot = nums[pivot_index]
            nums[pivot_index], nums[right] = nums[right], nums[pivot_index]
            store_index = left

            for i in range(left, right):
                if nums[i] < pivot:
                    nums[i], nums[store_index] = nums[store_index], nums[i]
                    store_index += 1

            nums[right], nums[store_index] = nums[store_index], nums[right]
            return store_index

        def quickselect(left, right, k_smallest):
            if left == right:
                return nums[left]

            pivot_index = random.randint(left, right)
            pivot_index = partition(left, right, pivot_index)

            if k_smallest == pivot_index:
                return nums[k_smallest]
            elif k_smallest < pivot_index:
                return quickselect(left, pivot_index - 1, k_smallest)
            else:
                return quickselect(pivot_index + 1, right, k_smallest)

        # Convert kth largest to kth smallest, where k = len(nums) - k
        k_smallest = len(nums) - k
        return quickselect(0, len(nums) - 1, k_smallest)
        '''

        '''
        nums.sort(reverse = True)
        return nums[k-1]
        '''


class Solution:
    def findPeakElement(self, nums: List[int]) -> int:

        left, right = 0, len(nums) - 1

        while left < right:
            mid = left + (right - left) // 2

            # Check if the middle element is greater than its neighbors
            if nums[mid] > nums[mid + 1]:
                right = mid
            else:
                left = mid + 1

        return left

        '''
        maxn = max(nums)

        for i, num in enumerate(nums):

            if num == maxn:
                return i
        '''


class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:

        def findLeft(nums, target):
            left, right = 0, len(nums) - 1
            while left < right:
                mid = left + (right - left) // 2
                if nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid
            if nums[left] == target:
                return left
            return -1

        def findRight(nums, target):
            left, right = 0, len(nums) - 1
            while left < right:
                mid = left + (right - left) // 2 + 1  # Adjusted the mid calculation
                if nums[mid] > target:
                    right = mid - 1
                else:
                    left = mid
            if nums[right] == target:
                return right
            return -1

        if nums == []:
            return [-1, -1]
        else:
            left = findLeft(nums, target)
            if left == -1:
                return [-1, -1]
            right = findRight(nums, target)

            return [left, right]

        '''
        start = -1
        count = counter(nums)
        for i, num in enumerate(nums):
            if num == target and start == -1:
                start = i
        '''


class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if not intervals:
            return []

        # Sort intervals by their start times
        intervals.sort(key=lambda x: x[0])

        merged = [intervals[0]]

        for i in range(1, len(intervals)):
            current_interval = intervals[i]
            last_merged = merged[-1]

            if current_interval[0] <= last_merged[1]:  # Overlapping intervals
                last_merged[1] = max(last_merged[1], current_interval[1])  # Merge
            else:
                merged.append(current_interval)  # No overlap, add as a new interval

        return merged


class Solution:
    def search(self, nums: List[int], target: int) -> int:
        for i, num in enumerate(nums):
            if num == target:
                return i
        return -1
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == target:
                    return True
        return False
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        max_reach = 0  # Maximum index you can reach
        n = len(nums)

        for i in range(n):
            if i > max_reach:
                return False  # If you can't reach index i, return False
            max_reach = max(max_reach, i + nums[i])  # Update the maximum reach

            if max_reach >= n - 1:
                return True  # If you can reach the last index, return True

        return True  # If you've gone through the loop, you can reach the end

class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        # Create a 2D array to store the number of paths
        dp = [[1] * n for _ in range(m)]

        # Fill the array using dynamic programming
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

        return dp[m - 1][n - 1]
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if not nums:
            return 0

        n = len(nums)
        dp = [1] * n  # Initialize an array to store the length of the longest increasing subsequence

        for i in range(1, n):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)

        return max(dp)
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        # Initialize dp array with a value greater than the amount
        dp = [amount + 1] * (amount + 1)
        dp[0] = 0

        # Fill the dp array
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] = min(dp[i], dp[i - coin] + 1)

        return dp[amount] if dp[amount] <= amount else -1


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """

        def preorder(node):
            if not node:
                return 'None,'
            return str(node.val) + ',' + preorder(node.left) + preorder(node.right)

        return preorder(root)

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """

        def build_tree(nodes):
            if nodes[0] == 'None':
                nodes.pop(0)
                return None
            root = TreeNode(int(nodes[0]))
            nodes.pop(0)
            root.left = build_tree(nodes)
            root.right = build_tree(nodes)
            return root

        nodes = data.split(',')
        return build_tree(nodes)

# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# ans = deser.deserialize(ser.serialize(root))
class RandomizedSet:

    def __init__(self):
        self.data = {}  # Dictionary to store (element: index) pairs
        self.elements = []  # List to store the elements

    def insert(self, val: int) -> bool:
        if val in self.data:
            return False
        self.data[val] = len(self.elements)
        self.elements.append(val)
        return True

    def remove(self, val: int) -> bool:
        if val in self.data:
            # Move the last element to the position of the element to be removed
            last_element = self.elements[-1]
            idx = self.data[val]
            self.elements[idx] = last_element
            self.data[last_element] = idx

            # Remove the last element and the element to be removed
            self.elements.pop()
            del self.data[val]
            return True
        return False

    def getRandom(self) -> int:
        return random.choice(self.elements)


# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()
class Solution:
    def isHappy(self, n: int) -> bool:
        def get_next(n):
            total_sum = 0
            while n > 0:
                n, digit = divmod(n, 10)
                total_sum += digit ** 2
            return total_sum

        seen = set()
        while n != 1 and n not in seen:
            seen.add(n)
            n = get_next(n)

        return n == 1
class Solution:
    def trailingZeroes(self, n: int) -> int:
        count = 0
        while n >= 5:
            n //= 5
            count += n
        return count
class Solution:
    def titleToNumber(self, columnTitle: str) -> int:
        columnNumber = 0
        for char in columnTitle:
            columnNumber = columnNumber * 26 + (ord(char) - ord('A') + 1)
        return columnNumber
class Solution:
    def myPow(self, x: float, n: int) -> float:
        return x ** n
class Solution:
    def mySqrt(self, x: int) -> int:
        return int(x ** (0.5))
class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        # return floor(dividend/divisor)
         # Constants
        INT_MAX = 2**31 - 1
        INT_MIN = -2**31

        if divisor == 0:
            return INT_MAX

        if dividend == INT_MIN and divisor == -1:
            return INT_MAX

        sign = -1 if (dividend < 0) ^ (divisor < 0) else 1
        dividend, divisor = abs(dividend), abs(divisor)

        result = 0
        while dividend >= divisor:
            temp, multiple = divisor, 1
            while dividend >= (temp << 1):
                temp <<= 1
                multiple <<= 1
            dividend -= temp
            result += multiple

        return result * sign
class Solution:
    def fractionToDecimal(self, numerator: int, denominator: int) -> str:
        # return str(numerator/denominator)
        if numerator == 0:
            return "0"

        result = []

        if (numerator < 0) ^ (denominator < 0):
            result.append("-")

        numerator = abs(numerator)
        denominator = abs(denominator)

        result.append(str(numerator // denominator))
        remainder = numerator % denominator

        if remainder == 0:
            return "".join(result)

        result.append(".")
        seen = {remainder: len(result)}

        while remainder != 0:
            numerator, remainder = divmod(remainder * 10, denominator)
            result.append(str(numerator))

            if remainder in seen:
                index = seen[remainder]
                result.insert(index, "(")
                result.append(")")
                break
            else:
                seen[remainder] = len(result)

        return "".join(result)

from collections import Counter
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        if n == 0:
            return len(tasks)
        task_count = list(Counter(tasks).values())
        max_freq = max(task_count)
        max_freq_count = task_count.count(max_freq)
        return max(len(tasks), (max_freq - 1) * (n + 1) + max_freq_count)
from collections import Counter
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        frequency_count = Counter(nums)
        most_common = frequency_count.most_common(1)
        # x, y = most_common
        return most_common[0][0]


class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stack = []

        for token in tokens:
            if token not in "+-*/":
                stack.append(int(token))
            else:
                b = stack.pop()
                a = stack.pop()
                if token == "+":
                    stack.append(a + b)
                elif token == "-":
                    stack.append(a - b)
                elif token == "*":
                    stack.append(a * b)
                else:
                    # Handle division by zero
                    stack.append(int(float(a) / b))

        return stack[0]
class Solution:
    def getSum(self, a: int, b: int) -> int:
        # return (a + b)
        mask = 0xFFFFFFFF  # 32-bit mask to get the 32 least significant bits
        while b != 0:
            # Calculate the carry
            carry = (a & b) & mask
            a = (a ^ b) & mask
            b = (carry << 1) & mask

        return a if a <= 0x7FFFFFFF else ~(a ^ mask)


class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = [1] * n

        # Calculate prefix products
        prefix_product = 1
        for i in range(n):
            result[i] *= prefix_product
            prefix_product *= nums[i]

        # Calculate suffix products
        suffix_product = 1
        for i in range(n - 1, -1, -1):
            result[i] *= suffix_product
            suffix_product *= nums[i]

        return result
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix:
            return []

        result = []
        left, right, top, bottom = 0, len(matrix[0]) - 1, 0, len(matrix) - 1

        while left <= right and top <= bottom:
            # Traverse top row
            for i in range(left, right + 1):
                result.append(matrix[top][i])
            top += 1

            # Traverse right column
            for i in range(top, bottom + 1):
                result.append(matrix[i][right])
            right -= 1

            if top <= bottom:
                # Traverse bottom row
                for i in range(right, left - 1, -1):
                    result.append(matrix[bottom][i])
                bottom -= 1

            if left <= right:
                # Traverse left column
                for i in range(bottom, top - 1, -1):
                    result.append(matrix[i][left])
                left += 1

        return result
class Solution:
    def fourSumCount(self, nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:
        sum_count = {}
        result = 0

        # Calculate the count of sums from the first two arrays (nums1 and nums2)
        for num1 in nums1:
            for num2 in nums2:
                sum_count[num1 + num2] = sum_count.get(num1 + num2, 0) + 1

        # Iterate through the other two arrays (nums3 and nums4) to find matching sums
        for num3 in nums3:
            for num4 in nums4:
                target = -(num3 + num4)
                if target in sum_count:
                    result += sum_count[target]

        return result


class Solution:
    def maxArea(self, height: List[int]) -> int:
        '''
        max_area = 0
        left = 0
        right = len(height) - 1

        while left < right:
            h = min(height[left], height[right])
            w = right - left
            area = h * w
            max_area = max(max_area, area)

            if height[left] < height[right]:
                left += 1
            else:
                right -= 1

        return max_area

        '''
        max_area = 0
        n = len(height)

        for i in range(n):
            for j in range(i + 1, n):
                h = min(height[i], height[j])
                w = j - i
                area = h * w
                max_area = max(max_area, area)

        return max_area
class Solution:
    def gameOfLife(self, board: List[List[int]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        def count_live_neighbors(row, col):
            live_neighbors = 0
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            for dr, dc in directions:
                r, c = row + dr, col + dc
                if 0 <= r < m and 0 <= c < n and (board[r][c] == 1 or board[r][c] == 2):
                    live_neighbors += 1
            return live_neighbors

        m, n = len(board), len(board[0])

        for row in range(m):
            for col in range(n):
                live_neighbors = count_live_neighbors(row, col)
                if board[row][col] == 1:
                    if live_neighbors < 2 or live_neighbors > 3:
                        board[row][col] = 2
                elif board[row][col] == 0:
                    if live_neighbors == 3:
                        board[row][col] = 3

        for row in range(m):
            for col in range(n):
                if board[row][col] == 2:
                    board[row][col] = 0
                elif board[row][col] == 3:
                    board[row][col] = 1

        """
        def count_live_neighbors(row, col):
            live_neighbors = 0
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            for dr, dc in directions:
                r, c = row + dr, col + dc
                if 0 <= r < m and 0 <= c < n and (board[r][c] == 1 or board[r][c] == -1):
                    live_neighbors += 1
            return live_neighbors

        m, n = len(board), len(board[0])
        updated_board = [[0] * n for _ in range(m)]

        for row in range(m):
            for col in range(n):
                live_neighbors = count_live_neighbors(row, col)
                if board[row][col] == 1:
                    if live_neighbors < 2 or live_neighbors > 3:
                        updated_board[row][col] = -1
                    else:
                        updated_board[row][col] = 1
                elif board[row][col] == 0 and live_neighbors == 3:
                    updated_board[row][col] = 1

        for row in range(m):
            for col in range(n):
                if updated_board[row][col] == -1:
                    updated_board[row][col] = 0

        for row in range(m):
            for col in range(n):
                board[row][col] = updated_board[row][col]
        """


class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        '''
        count = 1

        for num in range(len(nums) + 1):
            if count in nums:
                count += 1
            else:
                return count
        return 0
        '''

        n = len(nums)

        # First, handle the cases with negative numbers and zeros by setting them to n+1
        for i in range(n):
            if nums[i] <= 0:
                nums[i] = n + 1

        # Perform cyclic sort to place each number from 1 to n in its correct position
        for i in range(n):
            num = abs(nums[i])
            if 1 <= num <= n:
                nums[num - 1] = -abs(nums[num - 1])

        # The first index with a positive value corresponds to the smallest missing positive
        for i in range(n):
            if nums[i] > 0:
                return i + 1

        # If all numbers from 1 to n are present, return n+1
        return n + 1


class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        if not nums:
            return 0

        num_set = set(nums)
        longest_streak = 0

        for num in num_set:
            if num - 1 not in num_set:  # Check if the current number starts a new sequence
                current_num = num
                current_streak = 1

                while current_num + 1 in num_set:
                    current_num += 1
                    current_streak += 1

                longest_streak = max(longest_streak, current_streak)

        return longest_streak

        '''
        nums.sort()

        for i in range(len(nums)):
            if n[i] == 
        '''
from collections import Counter
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        count = Counter(nums)
        most_common = count.most_common(1)
        return most_common[0][0]
from collections import deque

class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if not nums:
            return []

        result = []
        dq = deque()

        for i in range(len(nums)):
            # Remove elements outside of the current window
            while dq and dq[0] < i - k + 1:
                dq.popleft()

            # Remove elements that are smaller and won't be needed
            while dq and nums[dq[-1]] < nums[i]:
                dq.pop()

            dq.append(i)

            # Add the maximum value in the current window to the result
            if i >= k - 1:
                result.append(nums[dq[0]])

        return result
from collections import Counter

class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if not s or not t:
            return ""

        # Initialize character counts for string t
        t_counts = Counter(t)

        left, right = 0, 0
        min_len = float('inf')
        min_window = ""

        # Count of characters from t that are still needed to be included
        required = len(t_counts)

        while right < len(s):
            # Expand the right pointer
            if s[right] in t_counts:
                t_counts[s[right]] -= 1
                if t_counts[s[right]] == 0:
                    required -= 1

            # Check if all characters from t are included
            while required == 0:
                if right - left + 1 < min_len:
                    min_len = right - left + 1
                    min_window = s[left:right+1]

                if s[left] in t_counts:
                    t_counts[s[left]] += 1
                    if t_counts[s[left]] > 0:
                        required += 1
                left += 1

            right += 1

        return min_window
import heapq

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        min_heap = []

        # Add the first element from each linked list to the min-heap
        for i, lst in enumerate(lists):
            if lst:
                heapq.heappush(min_heap, (lst.val, i))
                lists[i] = lst.next

        dummy = ListNode()
        current = dummy

        while min_heap:
            val, i = heapq.heappop(min_heap)
            current.next = ListNode(val)
            current = current.next

            if lists[i]:
                heapq.heappush(min_heap, (lists[i].val, i))
                lists[i] = lists[i].next

        return dummy.next
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head

        # Helper function to split the linked list into two halves
        def split(head):
            slow = head
            fast = head
            prev = None

            while fast and fast.next:
                prev = slow
                slow = slow.next
                fast = fast.next.next

            prev.next = None  # Split the list into two parts
            return head, slow

        # Helper function to merge two sorted linked lists
        def merge(l1, l2):
            dummy = ListNode(0)
            current = dummy

            while l1 and l2:
                if l1.val < l2.val:
                    current.next = l1
                    l1 = l1.next
                else:
                    current.next = l2
                    l2 = l2.next
                current = current.next

            if l1:
                current.next = l1
            if l2:
                current.next = l2

            return dummy.next

        # Merge sort
        left, right = split(head)
        left = self.sortList(left)
        right = self.sortList(right)
        return merge(left, right)
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""

class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if not head:
            return None

        # Step 1: Create a copy of each node and insert it next to the original node
        current = head
        while current:
            new_node = Node(current.val)
            new_node.next = current.next
            current.next = new_node
            current = new_node.next

        # Step 2: Update the random pointers of the copied nodes
        current = head
        while current:
            if current.random:
                current.next.random = current.random.next
            current = current.next.next

        # Step 3: Split the combined list into two separate lists
        original_head = head
        copied_head = head.next
        copied_current = copied_head
        while copied_current.next:
            original_head.next = copied_current.next
            original_head = original_head.next
            copied_current.next = original_head.next
            copied_current = copied_current.next

        original_head.next = None  # Set the last node in the original list to None
        return copied_head
from collections import deque
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        # Convert the wordList to a set for faster lookup
        wordSet = set(wordList)
        if endWord not in wordSet:
            return 0  # endWord is not in wordList, so no transformation sequence is possible

        queue = deque()
        queue.append((beginWord, 1))  # Start with the beginWord and a length of 1

        while queue:
            word, length = queue.popleft()
            if word == endWord:
                return length  # Found the endWord, return the length of the sequence

            for i in range(len(word)):
                for char in "abcdefghijklmnopqrstuvwxyz":
                    nextWord = word[:i] + char + word[i + 1:]
                    if nextWord in wordSet:
                        queue.append((nextWord, length + 1))
                        wordSet.remove(nextWord)  # Mark the word as visited to avoid loops

        return 0  # No transformation sequence is possible
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        if not board:
            return

        m, n = len(board), len(board[0])

        def dfs(row, col):
            if row < 0 or row >= m or col < 0 or col >= n or board[row][col] != 'O':
                return
            board[row][col] = 'T'
            dfs(row - 1, col)
            dfs(row + 1, col)
            dfs(row, col - 1)
            dfs(row, col + 1)

        # Traverse the first and last rows
        for col in range(n):
            dfs(0, col)
            dfs(m - 1, col)

        # Traverse the first and last columns
        for row in range(1, m - 1):
            dfs(row, 0)
            dfs(row, n - 1)

        # Convert 'O's to 'X' and 'T's back to 'O'
        for row in range(m):
            for col in range(n):
                if board[row][col] == 'O':
                    board[row][col] = 'X'
                elif board[row][col] == 'T':
                    board[row][col] = 'O'
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root:
            return None

        # Check if either of the target nodes is found
        if root == p or root == q:
            return root

        # Recursively search for the target nodes in the left and right subtrees
        left_ancestor = self.lowestCommonAncestor(root.left, p, q)
        right_ancestor = self.lowestCommonAncestor(root.right, p, q)

        # If both left and right ancestors are found, the current node is the LCA
        if left_ancestor and right_ancestor:
            return root

        # If only one ancestor is found, return it (not LCA yet)
        return left_ancestor or right_ancestor


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        # Initialize a variable to store the maximum path sum
        self.max_sum = float('-inf')

        def max_gain(node):
            if not node:
                return 0

            # Recursively compute the maximum path sum in the left and right subtrees
            left_gain = max(max_gain(node.left), 0)
            right_gain = max(max_gain(node.right), 0)

            # Update the maximum path sum considering the current node as the highest point
            self.max_sum = max(self.max_sum, left_gain + right_gain + node.val)

            # Return the maximum gain for the current subtree rooted at 'node'
            return max(left_gain, right_gain) + node.val

        # Start the recursive function from the root of the tree
        max_gain(root)

        return self.max_sum
class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        n = len(isConnected)
        visited = [False] * n
        provinces = 0

        def dfs(city):
            visited[city] = True
            for neighbor in range(n):
                if isConnected[city][neighbor] == 1 and not visited[neighbor]:
                    dfs(neighbor)

        for city in range(n):
            if not visited[city]:
                provinces += 1
                dfs(city)

        return provinces
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # Create an adjacency list to represent the graph
        graph = [[] for _ in range(numCourses)]
        for course, prereq in prerequisites:
            graph[course].append(prereq)

        def hasCycle(course):
            # 0 represents not visited, 1 represents visiting, 2 represents visited
            if visited[course] == 1:
                return True
            if visited[course] == 2:
                return False

            visited[course] = 1
            for neighbor in graph[course]:
                if hasCycle(neighbor):
                    return True
            visited[course] = 2
            return False

        visited = [0] * numCourses

        for course in range(numCourses):
            if visited[course] == 0:
                if hasCycle(course):
                    return False

        return True
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        # Create an adjacency list to represent the graph
        graph = [[] for _ in range(numCourses)]
        for course, prereq in prerequisites:
            graph[course].append(prereq)

        def dfs(course):
            # Mark the course as visited
            visited[course] = 1
            for neighbor in graph[course]:
                if visited[neighbor] == 0:
                    if not dfs(neighbor):
                        return False
                elif visited[neighbor] == 1:
                    return False
            # Mark the course as done
            visited[course] = 2
            result.append(course)
            return True

        visited = [0] * numCourses
        result = []

        for course in range(numCourses):
            if visited[course] == 0:
                if not dfs(course):
                    return []

        return result[:]  # Reverse the result to get the correct topological order
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        if not matrix:
            return 0

        m, n = len(matrix), len(matrix[0])
        memo = [[0] * n for _ in range(m)]
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        def dfs(x, y):
            if memo[x][y] != 0:
                return memo[x][y]

            max_length = 1
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < m and 0 <= new_y < n and matrix[new_x][new_y] > matrix[x][y]:
                    length = 1 + dfs(new_x, new_y)
                    max_length = max(max_length, length)

            memo[x][y] = max_length
            return max_length

        max_path_length = 0
        for i in range(m):
            for j in range(n):
                max_path_length = max(max_path_length, dfs(i, j))

        return max_path_length


class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        counts = [0] * len(nums)
        sorted_nums = []

        for i, num in enumerate(reversed(nums)):
            index = bisect_left(sorted_nums, num)
            counts[len(nums) - 1 - i] = index
            sorted_nums.insert(index, num)

        return counts
        '''
        sorted_nums = sorted(enumerate(nums), key=lambda x: x[1])
        result = [0] * len(nums)
        sorted_indices = [index for index, _ in sorted_nums]
        bit = [0] * (len(nums) + 1)

        def update(index):
            while index <= len(nums):
                bit[index] += 1
                index += index & -index

        def query(index):
            count = 0
            while index:
                count += bit[index]
                index -= index & -index
            return count

        for i, index in enumerate(sorted_indices):
            result[index] = query(i)
            update(i + 1)

        return result
        '''
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        def is_palindrome(substring):
            return substring == substring[::-1]

        def backtrack(start, path):
            if start == len(s):
                partitions.append(path[:])
                return

            for end in range(start + 1, len(s) + 1):
                if is_palindrome(s[start:end]):
                    path.append(s[start:end])
                    backtrack(end, path)
                    path.pop()

        partitions = []
        backtrack(0, [])
        return partitions


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True


class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        def backtrack(node, r, c, path):
            letter = board[r][c]
            curr_node = node.children.get(letter)

            if not curr_node:
                return

            path.append(letter)
            board[r][c] = "#"  # Mark visited

            if curr_node.is_end_of_word:
                found_words.add("".join(path))

            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and board[nr][nc] != "#" and board[nr][nc] in curr_node.children:
                    backtrack(curr_node, nr, nc, path)

            path.pop()
            board[r][c] = letter  # Restore the board state

        m, n = len(board), len(board[0])
        trie = Trie()
        for word in words:
            trie.insert(word)

        found_words = set()
        for r in range(m):
            for c in range(n):
                if board[r][c] in trie.root.children:
                    backtrack(trie.root, r, c, [])

        return list(found_words)


class Solution:
    def removeInvalidParentheses(self, s: str) -> List[str]:
        def is_valid(s):
            balance = 0
            for char in s:
                if char == "(":
                    balance += 1
                elif char == ")":
                    balance -= 1
                if balance < 0:
                    return False
            return balance == 0

        def dfs(index, left_count, right_count, left_to_remove, right_to_remove, current):
            if index == len(s):
                if left_to_remove == 0 and right_to_remove == 0 and is_valid(current):
                    valid_parentheses.add(current)
                return

            char = s[index]
            if char == "(" and left_to_remove > 0:
                dfs(index + 1, left_count, right_count, left_to_remove - 1, right_to_remove, current)
            elif char == ")" and right_to_remove > 0:
                dfs(index + 1, left_count, right_count, left_to_remove, right_to_remove - 1, current)

            current += char
            if char != "(" and char != ")":
                dfs(index + 1, left_count, right_count, left_to_remove, right_to_remove, current)
            elif char == "(":
                dfs(index + 1, left_count + 1, right_count, left_to_remove, right_to_remove, current)
            elif char == ")" and left_count > right_count:
                dfs(index + 1, left_count, right_count + 1, left_to_remove, right_to_remove, current)

        left_count, right_count = 0, 0
        for char in s:
            if char == "(":
                left_count += 1
            elif char == ")":
                if left_count > 0:
                    left_count -= 1
                else:
                    right_count += 1

        valid_parentheses = set()
        dfs(0, 0, 0, left_count, right_count, "")

        return list(valid_parentheses)


class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True

        for j in range(1, n + 1):
            if p[j - 1] == '*':
                dp[0][j] = dp[0][j - 1]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == s[i - 1] or p[j - 1] == '?':
                    dp[i][j] = dp[i - 1][j - 1]
                elif p[j - 1] == '*':
                    dp[i][j] = dp[i - 1][j] or dp[i][j - 1]

        return dp[m][n]
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True

        for i in range(m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    dp[i][j] = dp[i][j - 2] or (i > 0 and (s[i - 1] == p[j - 2] or p[j - 2] == '.') and dp[i - 1][j])
                else:
                    dp[i][j] = dp[i - 1][j - 1] and (s[i - 1] == p[j - 1] or p[j - 1] == '.')

        return dp[m][n]
class Solution:
    def wiggleSort(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)

        # Find the median element
        median = self.findKthLargest(nums, (n + 1) // 2)

        # Index-rewiring function
        index = lambda i: (1 + 2 * i) % (n | 1)

        # Dutch National Flag algorithm for wiggling
        low, mid, high = 0, 0, n - 1
        while mid <= high:
            if nums[index(mid)] > median:
                nums[index(low)], nums[index(mid)] = nums[index(mid)], nums[index(low)]
                low += 1
                mid += 1
            elif nums[index(mid)] < median:
                nums[index(high)], nums[index(mid)] = nums[index(mid)], nums[index(high)]
                high -= 1
            else:
                mid += 1

    def findKthLargest(self, nums, k):
        return sorted(nums)[-k]
        '''
        # Sort the array
        nums.sort()
        n = len(nums)
        # Calculate mid index
        mid = (n - 1) // 2
        # Create two pointers, one for smaller values and one for larger values
        i, j = 0, mid + 1
        result = [0] * n
        k = 0

        # Interleave the values, first smaller, then larger
        while j < n:
            result[k] = nums[mid - i]
            result[k + 1] = nums[n - 1 - i]
            i += 1
            j += 1
            k += 2

        # If the array has an odd length, add the middle element
        if i <= mid:
            result[-1] = nums[mid - i]

        # Copy the result back to the original array
        for i in range(n):
            nums[i] = result[i]
        '''
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        list = []
        # list.append(matrix[0][:])
        # list.append(matrix[:][0])
        # return list[k]
        flattened = [elem for row in matrix for elem in row]
        flattened.sort()
        return flattened[k - 1]
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        list1 = [i for i in nums1]
        list2 = [j for j in nums2]
        listf = sorted(list1 + list2)  # Combine and sort the lists
        n = len(listf)
        if n % 2 == 0:
            # If the length is even, return the average of the middle two elements
            mid = n // 2
            return (listf[mid - 1] + listf[mid]) / 2
        else:
            # If the length is odd, return the middle element
            return listf[n // 2]
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        if not nums:
            return 0

        max_product = nums[0]
        min_product = nums[0]
        result = max_product

        for i in range(1, len(nums)):
            if nums[i] < 0:
                max_product, min_product = min_product, max_product

            max_product = max(nums[i], max_product * nums[i])
            min_product = min(nums[i], min_product * nums[i])

            result = max(result, max_product)

        return result

class Solution:
    def numDecodings(self, s: str) -> int:
        n = len(s)
        if n == 0:
            return 0

        dp = [0] * (n + 1)
        dp[0] = 1
        dp[1] = 1 if s[0] != '0' else 0

        for i in range(2, n + 1):
            if s[i - 1] != '0':
                dp[i] += dp[i - 1]
            if '10' <= s[i - 2:i] <= '26':
                dp[i] += dp[i - 2]

        return dp[n]


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0

        n = len(prices)
        buy = [0] * n
        sell = [0] * n
        cooldown = [0] * n

        buy[0] = -prices[0]  # Buy the stock on the first day
        for i in range(1, n):
            # You can either continue to be in the cooldown state or buy a stock on the ith day
            buy[i] = max(buy[i - 1], cooldown[i - 1] - prices[i])
            # You can either continue to be in the sell state or sell the stock on the ith day
            sell[i] = max(sell[i - 1], buy[i - 1] + prices[i])
            # You have to be in the cooldown state after selling the stock
            cooldown[i] = max(cooldown[i - 1], sell[i - 1])

        # The maximum profit will be in the sell state, as you must sell the stock to realize the profit
        return sell[-1]


class Solution:
    def numSquares(self, n: int) -> int:
        if n <= 0:
            return 0

        # Create a list to store the number of perfect squares required to sum to each number from 1 to n
        dp = [float('inf')] * (n + 1)

        # The number of perfect squares required to sum to 0 is 0
        dp[0] = 0

        # Generate a list of perfect squares up to n
        perfect_squares = [i * i for i in range(1, int(n ** 0.5) + 1)]

        # Update the dp list for each number from 1 to n
        for i in range(1, n + 1):
            for square in perfect_squares:
                if i < square:
                    break
                dp[i] = min(dp[i], dp[i - square] + 1)

        return dp[n]

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        # Create a set from the wordDict for faster word lookup
        word_set = set(wordDict)

        # Create a dynamic programming table to keep track of valid segments
        dp = [False] * (len(s) + 1)
        dp[0] = True  # An empty string is a valid segment

        # Iterate through each position in the string
        for i in range(1, len(s) + 1):
            # Check all substrings ending at this position
            for j in range(i):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break

        return dp[len(s)]
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        # Create a set from the wordDict for faster word lookup
        word_set = set(wordDict)

        # Create a dynamic programming table to keep track of valid segments
        dp = [False] * (len(s) + 1)
        dp[0] = True  # An empty string is a valid segment

        # Calculate dp table
        for i in range(1, len(s) + 1):
            for j in range(i):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break

        def backtrack(start, path):
            if start == len(s):
                sentences.append(" ".join(path))
                return
            for end in range(start + 1, len(s) + 1):
                if dp[end] and s[start:end] in word_set:
                    backtrack(end, path + [s[start:end]])

        sentences = []
        if dp[len(s)]:
            backtrack(0, [])

        return sentences


class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        n = len(nums)

        # Pad the nums array with 1 on both sides to handle edge cases
        nums = [1] + nums + [1]

        # Initialize a dynamic programming table dp with zeros
        dp = [[0] * (n + 2) for _ in range(n + 2)]

        # Perform bottom-up dynamic programming
        for length in range(1, n + 1):
            for i in range(1, n - length + 2):
                j = i + length - 1
                for k in range(i, j + 1):
                    dp[i][j] = max(dp[i][j], dp[i][k - 1] + nums[i - 1] * nums[k] * nums[j + 1] + dp[k + 1][j])

        # The result is stored in dp[1][n]
        return dp[1][n]
class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.order = DoublyLinkedList()


    def get(self, key: int) -> int:
        if key in self.cache:
            self.order.move_to_front(self.cache[key])
            return self.cache[key].value
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache[key].value = value
            self.order.move_to_front(self.cache[key])
        else:
            if len(self.cache) >= self.capacity:
                removed = self.order.remove_last()
                del self.cache[removed.key]
            new_node = Node(key, value)
            self.cache[key] = new_node
            self.order.add_to_front(new_node)
class Node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class DoublyLinkedList:
    def __init__(self):
        self.head = Node(-1, -1)
        self.tail = Node(-1, -1)
        self.head.next = self.tail
        self.tail.prev = self.head

    def add_to_front(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def move_to_front(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev
        self.add_to_front(node)

    def remove_last(self):
        last_node = self.tail.prev
        self.tail.prev = last_node.prev
        last_node.prev.next = self.tail
        return last_node

# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
class Trie:

    def __init__(self):
        self.trie = {}


    def insert(self, word: str) -> None:
        node = self.trie
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node['end'] = True

    def search(self, word: str) -> bool:
        node = self.trie
        for char in word:
            if char not in node:
                return False
            node = node[char]
        return 'end' in node


    def startsWith(self, prefix: str) -> bool:
        node = self.trie
        for char in prefix:
            if char not in node:
                return False
            node = node[char]
        return True


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)
class Solution:
    def largestNumber(self, nums: List[int]) -> str:
        def custom_compare(a, b):
            return int(b + a) - int(a + b)

        nums = sorted(map(str, nums), key=functools.cmp_to_key(custom_compare))
        result = ''.join(nums)
        return str(int(result))  # Handle cases with leading zeros
class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        if len(points) <= 2:
            return len(points)

        max_points = 2  # At least 2 points are needed to form a line

        for i in range(len(points)):
            slopes = defaultdict(int)
            duplicates = 0
            local_max = 1

            for j in range(i + 1, len(points)):
                x1, y1 = points[i]
                x2, y2 = points[j]

                if x1 == x2 and y1 == y2:
                    duplicates += 1
                elif x1 == x2:
                    slope = float('inf')
                    slopes[slope] += 1
                    local_max = max(local_max, slopes[slope])
                else:
                    slope = (y2 - y1) / (x2 - x1)
                    slopes[slope] += 1
                    local_max = max(local_max, slopes[slope])

            max_points = max(max_points, local_max + duplicates + 1)

        return max_points


class Solution:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        people.sort(key=lambda x: (-x[0], x[1]))

        result = []
        for person in people:
            result.insert(person[1], person)

        return result


class Solution:
    def trap(self, height: List[int]) -> int:
        left, right = 0, len(height) - 1
        left_max = right_max = 0
        result = 0

        while left < right:
            if height[left] <= height[right]:
                left_max = max(left_max, height[left])
                result += max(0, left_max - height[left])
                left += 1
            else:
                right_max = max(right_max, height[right])
                result += max(0, right_max - height[right])
                right -= 1

        return result


class Solution:
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        # Create a list of critical points with heights (positive for left edges, negative for right edges)
        points = [(l, -h, r) for l, r, h in buildings]
        points += [(r, 0, 0) for _, r, _ in buildings]
        points.sort()

        result = []
        max_heap = [(0, float('inf'))]  # (height, right end)

        for x, neg_h, r in points:
            while max_heap[0][1] <= x:  # Remove buildings that have reached their end
                heapq.heappop(max_heap)

            if neg_h:  # If it's a left edge
                heapq.heappush(max_heap, (neg_h, r))

            # The maximum height in the heap represents the highest building at this point
            if not result or result[-1][1] != -max_heap[0][0]:
                result.append([x, -max_heap[0][0]])

        return result


'''
    def merge(self, left, right):
        result = []
        i, j, h1, h2 = 0, 0, 0, 0
        while i < len(left) and j < len(right):
            x1, h1 = left[i]
            x2, h2 = right[j]
            x = min(x1, x2)
            if x1 < x2:
                i += 1
            else:
                j += 1
            min_h = min(h1, h2)  # Corrected here
            if not result or min_h != result[-1][1]:
                result.append([x, min_h])  # Corrected here
        result += left[i:]
        result += right[j:]
        return result




        if not buildings:
                return []

        if len(buildings) == 1:
            x1, x2, h = buildings[0]
            return [[x1, h], [x2, 0]]

        mid = len(buildings) // 2
        left_buildings = buildings[:mid]
        right_buildings = buildings[mid:]

        left_skyline = self.getSkyline(left_buildings)
        right_skyline = self.getSkyline(right_buildings)

        return self.merge(left_skyline, right_skyline)
    '''
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        stack = []  # Use a stack to keep track of indices
        max_area = 0

        for i, h in enumerate(heights):
            while stack and h < heights[stack[-1]]:
                height = heights[stack.pop()]
                width = i if not stack else i - stack[-1] - 1
                max_area = max(max_area, height * width)
            stack.append(i)

        # After processing all elements, calculate areas for remaining elements in the stack
        while stack:
            height = heights[stack.pop()]
            width = len(heights) if not stack else len(heights) - stack[-1] - 1
            max_area = max(max_area, height * width)

        return max_area
class MedianFinder:

    def __init__(self):
        self.max_heap = []  # To store the left half (smaller elements)
        self.min_heap = []  # To store the right half (larger elements)


    def addNum(self, num: int) -> None:
        # First, add the number to the left (max-heap)
        # Then, rebalance the heaps
        heapq.heappush(self.max_heap, -num)
        heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))

        # Ensure the max-heap has at most one more element than the min-heap
        if len(self.max_heap) < len(self.min_heap):
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))


    def findMedian(self) -> float:
        if len(self.max_heap) == len(self.min_heap):
            # Even number of elements, so return the average of the two middle elements
            return (self.min_heap[0] - self.max_heap[0]) / 2
        else:
            # Odd number of elements, so the middle element from the max-heap is the median
            return -self.max_heap[0]


# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()

# """
# This is the interface that allows for creating nested lists.
# You should not implement it, or speculate about its implementation
# """
# class NestedInteger:
#    def isInteger(self) -> bool:
#        """
#        @return True if this NestedInteger holds a single integer, rather than a nested list.
#        """
#
#    def getInteger(self) -> int:
#        """
#        @return the single integer that this NestedInteger holds, if it holds a single integer
#        Return None if this NestedInteger holds a nested list
#        """
#
#    def getList(self) -> [NestedInteger]:
#        """
#        @return the nested list that this NestedInteger holds, if it holds a nested list
#        Return None if this NestedInteger holds a single integer
#        """

class NestedIterator:
    def __init__(self, nestedList: [NestedInteger]):
        self.flatten_list = []
        self.index = 0

        def flatten(nested):
            for item in nested:
                if item.isInteger():
                    self.flatten_list.append(item.getInteger())
                else:
                    flatten(item.getList())

        flatten(nestedList)

    def next(self) -> int:
        if self.hasNext():
            val = self.flatten_list[self.index]
            self.index += 1
            return val

    def hasNext(self) -> bool:
        return self.index < len(self.flatten_list)

# Your NestedIterator object will be instantiated and called as such:
# i, v = NestedIterator(nestedList), []
# while i.hasNext(): v.append(i.next())
