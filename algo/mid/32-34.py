def arrayOfProducts(array):
    # Write your code here.
    left_dp = []
    right_dp = []
    left = right = 1
    ans = []
    for num in array:
        left_dp.append(left)
        left *= num
    for num in reversed(array):
        right_dp.append(right)
        right *= num
    for i in range(len(array)):
        ans.append(right_dp[len(array) - i - 1] * left_dp[i])
    return ans


def firstDuplicateValue(array):
    # Write your code here.
    for value in array:
        absValue = abs(value)
        if array[absValue - 1] < 0:
            return absValue
        array[absValue - 1] *= -1
    return -1


def mergeOverlappingIntervals(intervals):
    # Write your code here.
    intervals = sorted(intervals, key=lambda x: x[0])
    ans = []
    cur_interval = intervals[0]
    for interval in intervals:
        if cur_interval[1] < interval[0]:
            ans.append(cur_interval)
            cur_interval = interval
        else:
            cur_interval[1] = max(cur_interval[1], interval[1])
    ans.append(cur_interval)
    return ans
