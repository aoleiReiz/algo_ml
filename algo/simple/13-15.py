# This is an input class. Do not edit.
class LinkedList:
    def __init__(self, value):
        self.value = value
        self.next = None


def removeDuplicatesFromLinkedList(linkedList):
    # Write your code here.
    p = linkedList
    while p and p.next:
        if p.value == p.next.value:
            p.next = p.next.next
        else:
            p = p.next
    return linkedList


def getNthFib(n):
    # Write your code here.
    if n == 1:
        return 0
    prev = 0
    curr = 1
    for i in range(2, n):
        prev, curr = curr, prev + curr
    return curr


def productSum(array):
    # Write your code here.
    def helper(arr, depth):
        total = 0
        for a in arr:
            if isinstance(a, list):
                total += depth * helper(a, depth + 1)
            else:
                total += a
        return total
    return helper(array, 2)


nums = [5, 2, [7, -1], 3, [6, [-13, 8], 4]]
print(productSum(nums))

