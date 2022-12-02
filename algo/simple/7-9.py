class BinaryTree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


def branchSums(root):
    # Write your code here.
    def helper(node: BinaryTree, cur_branch_sum, ans):
        if node is None:
            return
        if node.left is None and node.right is None:
            ans.append(cur_branch_sum + node.value)
            return
        cur_branch_sum += node.value
        if node.left is not None:
            helper(node.left, cur_branch_sum, ans)
        if node.right is not None:
            helper(node.right, cur_branch_sum, ans)

    a = []
    helper(root, 0, a)
    return a


def nodeDepths(root: BinaryTree):
    def helper(node: BinaryTree, depth):
        if node is None:
            return 0
        left_depth = helper(node.left, depth + 1)
        right_depth = helper(node.right, depth + 1)
        return depth + left_depth + right_depth
    return helper(root, 0)


# and methods to the class.
class Node:
    def __init__(self, name):
        self.children = []
        self.name = name

    def addChild(self, name):
        self.children.append(Node(name))
        return self

    def depthFirstSearch(self, array):
        # Write your code here.
        array.append(self.name)
        for child in self.children:
            child.depthFirstSearch(array)
        return array

