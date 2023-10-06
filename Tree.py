class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


class BinaryTree:
    def __init__(self, root):
        self.root = TreeNode(root)

    def preorder_traversal(self, start, traversal):
        if start:
            traversal.append(start.value)
            traversal = self.preorder_traversal(start.left, traversal)
            traversal = self.preorder_traversal(start.right, traversal)
        return traversal

    def inorder_traversal(self, start, traversal):
        if start:
            traversal = self.inorder_traversal(start.left, traversal)
            traversal.append(start.value)
            traversal = self.inorder_traversal(start.right, traversal)
        return traversal

    def postorder_traversal(self, start, traversal):
        if start:
            traversal = self.postorder_traversal(start.left, traversal)
            traversal = self.postorder_traversal(start.right, traversal)
            traversal.append(start.value)
        return traversal


# Create a binary tree
tree = BinaryTree(1)
tree.root.left = TreeNode(2)
tree.root.right = TreeNode(3)
tree.root.left.left = TreeNode(4)
tree.root.left.right = TreeNode(5)

# Perform tree traversals
print("Preorder Traversal:", tree.preorder_traversal(tree.root, []))
print("Inorder Traversal:", tree.inorder_traversal(tree.root, []))
print("Postorder Traversal:", tree.postorder_traversal(tree.root, []))
