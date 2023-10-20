from typing import Optional
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        # Base case: an empty tree has a depth of 0
        if root is None:
            return 0

        # Recursively calculate the depth of left and right subtrees
        left_depth = self.maxDepth(root.left)
        right_depth = self.maxDepth(root.right)

        # Return the maximum depth of left or right subtree plus 1 for the current node
        return max(left_depth, right_depth) + 1

# Example input tree: [3,9,20,null,null,15,7]
root = TreeNode(3)
root.left = TreeNode(9)
root.right = TreeNode(20)
root.right.left = TreeNode(15)
root.right.right = TreeNode(7)

# Create an instance of the Solution class
solution = Solution()

# Calculate the maximum depth of the tree
max_depth = solution.maxDepth(root)

# Print the result
print("Maximum depth of the tree:", max_depth)
