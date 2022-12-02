def twoNumberSum(array, targetSum):
    """space O(n), time: O(n)"""
    s = set()
    for num in array:
        rem = targetSum - num
        if rem in s:
            return [rem, num]
        else:
            s.add(num)
    return []

def twoNumberSum2(array, targetSum):
    """space O(1), time: O(nlogn)"""
    array = sorted(array)
    i = 0
    j = len(array) - 1
    while i < j:
        if array[i] + array[j] == targetSum:
            return [array[i], array[j]]
        elif array[i] + array[j] > targetSum:
            j -= 1
        else:
            i += 1
    return []


def isValidSubsequence(array, sequence):
    arrIdx = 0
    seqIdx = 0
    while arrIdx < len(array) and seqIdx < len(sequence):
        if array[arrIdx] == sequence[seqIdx]:
            seqIdx += 1
        arrIdx += 1
    return seqIdx == len(sequence)


def sortedSquaredArray(array):
    small_idx = 0
    large_idx = len(array) - 1
    ans = [0 for _ in range(len(array))]
    for idx in reversed(range(len(array))):
        left_value = array[small_idx] ** 2
        right_value = array[large_idx] ** 2
        if left_value >= right_value:
            ans[idx] = left_value
            small_idx += 1
        else:
            ans[idx] = right_value
            large_idx -= 1
    return ans

