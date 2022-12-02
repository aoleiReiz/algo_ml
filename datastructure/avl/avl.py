class Node:

    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.height = 1
        self.left = None
        self.right = None


class AVL:

    def __init__(self):
        self._root = None
        self._size = 0

    def get_size(self):
        return self._size

    def is_empty(self):
        return self._size == 0

    def _get_height(self, node: Node):
        """获得节点node的高度"""
        if node is None:
            return 0
        else:
            return node.height

    def _get_balance_factor(self, node: Node):
        """获得节点的平衡因子"""
        if node is None:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)

    def add(self, key, value):
        self._root = self._add(self._root, key, value)

    def _add(self, node: Node, key, value):
        if node is None:
            self._size += 1
            return Node(key, value)
        if key < node.key:
            node.left = self._add(node.left, key, value)
        elif key > node.key:
            node.right = self._add(node.right, key, value)
        else:
            node.value = value
        # 更新height
        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))
        # 计算平衡因子
        balance_factor = self._get_balance_factor(node)
        if abs(balance_factor) > 1:
            print(f"unbalanced : {balance_factor}")
        return node

    def get_node(self, node:Node, key):
        """返回以node为根节点的二分搜索树中，key所在的节点"""
        if node is None:
            return None
        if key == node.key:
            return node
        elif key < node.key:
            return self.get_node(node.left, key)
        else:
            return self.get_node(node.right, key)

    def contains(self, key):
        return self.get_node(self._root, key) is not None

    def get(self, key):
        node = self.get_node(self._root, key)
        return None if node is None else node.value

    def set(self, key, value):
        node = self.get_node(self._root, key)
        if node is not None:
            node.value = value

    def _minimum(self, node):
        if node.left is None:
            return node
        return self._minimum(node.left)

    def _remove_min(self, node):
        """
        删除掉以node为根的二分搜索树中的最小节点
        返回删除节点后新的二分搜索树的根
        """
        if node.left is None:
            right_node = node.right
            node.right = None
            self._size -= 1
            return right_node
        node.left = self._remove_min(node.left)
        return node

    def remove(self, key):
        node = self.get_node(self._root, key)
        if node is not None:
            self._root = self._remove(self._root, key)
            return node.value
        return None

    def _remove(self, node:Node, key):
        if node is None:
            return None
        if key < node.key:
            node.left = self._remove(node.left, key)
            return node
        elif key > node.key:
            node.right = self._remove(node.right, key)
            return node
        else:
            if node.left is None:
                right_node = node.right
                node.right = None
                self._size -= 1
                return right_node
            if node.right is None:
                left_node = node.left
                node.left = None
                self._size -= 1
                return left_node
            # 待删除节点左右子树均不为空的情况
            successor = self._minimum(node.right)
            successor.right = self._remove_min(node.right)
            successor.left = node.left

            node.left = None
            node.right = None
            return successor


