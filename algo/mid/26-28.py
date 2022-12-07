def threeNumberSum(array, targetSum):
    def two_sum(arr, t):
        left = 0
        right = len(arr) - 1
        ret = []
        while left < right:
            s = arr[left] + arr[right]
            if s == t:
                ret.append([arr[left], arr[right]])
                left += 1
                right -= 1
            elif s > t:
                right -= 1
            else:
                left += 1
        return ret

    ans = []
    array = sorted(array)
    for idx in range(len(array) - 2):
        targetSumLeft = targetSum - array[idx]
        two_sum_result = two_sum(array[idx + 1:], targetSumLeft)
        for tsr in two_sum_result:
            ans.append([array[idx], * tsr])
    return ans


def smallestDifference(arrayOne, arrayTwo):
    # Write your code here.
    arrayOne = sorted(arrayOne)
    arrayTwo = sorted(arrayTwo)
    idx1 = idx2 = 0
    smallest = float("inf")
    ans = []
    while idx1 < len(arrayOne) and idx2 < len(arrayTwo):
        num1 = arrayOne[idx1]
        num2 = arrayTwo[idx2]
        current = abs(num1 - num2)
        if current < smallest:
            smallest = current
            ans = [num1, num2]
        if num1 > num2:
            idx2 += 1
        elif num1 < num2:
            idx1 += 1
        else:
            return [num1, num2]
    return ans


def moveElementToEnd(array, toMove):
    left = 0
    right = len(array) - 1
    while left < right:
        while right > left and array[right] == toMove:
            right -= 1
        if array[left] == toMove:
            array[left], array[right] = array[right], array[left]
        left += 1
    return array

a = [2, 1, 2, 2, 2, 3, 4, 2]
print(moveElementToEnd(a, 2))





