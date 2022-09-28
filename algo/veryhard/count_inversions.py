def countInversions(array):
    # Write your code here.
    count = 0
    for i in range(len(array)):
        for j in range(i+1, len(array)):
            if array[i] > array[j]:
                count += 1
    return count


def countInversions2(array):
    # Write your code here.
    return merge_sort_count_inversions(array, 0, len(array) - 1)


def merge_sort_count_inversions(array, left, right):
    if left >= right:
        return 0
    mid = (left + right) // 2
    left_count = merge_sort_count_inversions(array, left, mid)
    right_count = merge_sort_count_inversions(array, mid+1, right)
    merge_count = merge_inversion_count(array, left, right, mid)
    return left_count + right_count + merge_count


def merge_inversion_count(array, left, right, mid):
    aux = array[left: right + 1][:]
    i = left
    j = mid + 1
    k = left
    inversions = 0
    while i <= mid and j <= right:
        if aux[i - left] <= aux[j - left]:
            array[k] = aux[i - left]
            i += 1
            k += 1
        else:
            array[k] = aux[j - left]
            inversions += mid + 1 - i
            j += 1
            k += 1
    while i <= mid:
        array[k] = aux[i - left]
        i += 1
        k += 1
    while j <= right:
        array[k] = aux[j - left]
        j += 1
        k += 1
    return inversions



array = [2, 3, 3, 1, 9, 5, 6]
print(countInversions2(array))
print(array)

