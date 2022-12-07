def isMonotonic(array):
    # Write your code here.
    n = len(array)
    if n > 2:
        flag = array[0] - array[-1] > 0
        for i in range(n-1):
            if array[i] - array[i + 1] != 0 and (array[i] - array[i + 1] > 0) != flag:
                return False
    return True


def spiralTraverse(array):
    m = len(array)
    if m == 0:
        return []
    n = len(array[0])
    ans = []
    row_start = 0
    row_end = m - 1
    col_start = 0
    col_end = n - 1

    while row_start <= row_end and col_start <= col_end:
        for c in range(col_start, col_end + 1):
            ans.append(array[row_start][c])
        row_start += 1

        for r in range(row_start, row_end + 1):
            ans.append(array[r][col_end])
        col_end -= 1

        if row_start > row_end:
            break
        for c in reversed(range(col_start, col_end + 1)):
            ans.append(array[row_end][c])
        row_end -= 1

        if col_start > col_end:
            break
        for r in reversed(range(row_start, row_end + 1)):
            ans.append(array[r][col_start])
        col_start += 1
    return ans


def longestPeak(array):
    n = len(array)
    ans = 0
    i = 1
    while i < n - 1:
        is_peak = array[i - 1] < array[i] > array[i + 1]
        if not is_peak:
            i += 1
            continue
        left_idx = i - 2
        while left_idx >=0 and array[left_idx] < array[left_idx + 1]:
            left_idx -= 1
        right_idx = i + 2
        while right_idx < n and array[right_idx] < array[right_idx - 1]:
            right_idx += 1
        ans = max(ans, right_idx - left_idx - 1)
        i = right_idx
    return ans





