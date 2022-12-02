def tournamentWinner(competitions, results):
    score_dict = {}
    final_winner = ""
    max_score = -1
    for i, competition in enumerate(competitions):
        winner = competition[0] if results[i] == 1 else competition[1]
        if winner not in score_dict:
            score_dict[winner] = 3
        else:
            score_dict[winner] += 3
        if score_dict[winner] > max_score:
            max_score = score_dict[winner]
            final_winner = winner
    return final_winner


def nonConstructibleChange(coins):
    # Write your code here.
    coins = sorted(coins)
    res = 0
    for i, coin in enumerate(coins):
        if coin > res + 1:
            return res + 1
        res += coin
    return res + 1


def findClosestValueInBst(tree, target):
    def helper(node, t, closest):
        if node is None:
            return closest
        if abs(t - closest) > abs(node.value - t):
            closest = node.value
        if node.value > target:
            return helper(node.left, t, closest)
        elif node.value < target:
            return helper(node.right, t, closest)
        else:
            return closest
    return helper(tree, target, tree.value)


# This is the class of the input tree. Do not edit.
class BST:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
