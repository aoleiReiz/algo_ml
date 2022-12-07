def insertionSort(array):
    # Write your code here.
    for i in range(len(array)):
        e = array[i]
        j = i
        while j > 0 and array[j - 1] > e:
            array[j] = array[j-1]
            j -= 1
        array[j] = e
    return array


def selectionSort(array):
    n = len(array)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if array[j] < array[min_idx]:
                min_idx = j
        array[min_idx], array[i] = array[i], array[min_idx]
    return array


def isPalindrome(string):
    # Write your code here.
    left = 0
    right = len(string) - 1

    while left < right:
        if string[left] == string[right]:
            left += 1
            right -= 1
        else:
            return False
    return True

a = [3, 5, 2, 1, 4]
print(insertionSort(a))

