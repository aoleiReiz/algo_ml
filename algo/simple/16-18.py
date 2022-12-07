def binarySearch(array, target):
    # Write your code here.
    left = 0
    right = len(array) - 1
    while left <= right:
        mid = (left + right) // 2
        if array[mid] == target:
            return mid
        elif array[mid] > target:
            right = mid - 1
        else:
            left = mid + 1
    return -1


def findThreeLargestNumbers(array):
    # Write your code here.
    firstMax = float("-inf")
    secondMax = float("-inf")
    thirdMax = float("-inf")
    for num in array:
        if num > firstMax:
            temp1 = firstMax
            temp2 = secondMax
            firstMax = num
            secondMax = temp1
            thirdMax = temp2
        elif num > secondMax:
            temp2 = secondMax
            secondMax = num
            thirdMax = temp2
        elif num > thirdMax:
            thirdMax = num
    return [thirdMax, secondMax, firstMax]


def bubbleSort(array):
    # Write your code here.
    n = len(array)
    for i in range(n):
        flag = False
        for j in reversed(range(n-1)):
            if array[j + 1] < array[j]:
                array[j + 1], array[j] = array[j], array[j + 1]
                flag = True
        if not flag:
            break
    return array

