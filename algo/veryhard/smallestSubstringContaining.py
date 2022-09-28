
def build_char_count(string):
    char_count = {}
    for c in string:
        if c not in char_count:
            char_count[c] = 1
        else:
            char_count[c] += 1
    return char_count


def smallestSubstringContaining(bigString, smallString):
    small_string_char_count = build_char_count(smallString)
    large_string_char_count = {}
    left = 0
    right = 0
    valid = 0
    ans = ""
    while right < len(bigString):
        char = bigString[right]
        right += 1
        if char not in small_string_char_count:
            continue
        if char in large_string_char_count:
            large_string_char_count[char] += 1
        else:
            large_string_char_count[char] = 1
        if large_string_char_count[char] == small_string_char_count[char]:
            valid += 1
        while valid == len(small_string_char_count):
            if right - left < len(ans) or len(ans) == 0:
                ans = bigString[left: right]
            char = bigString[left]
            left += 1
            if char in large_string_char_count:
                large_string_char_count[char] -= 1
                if large_string_char_count[char] < small_string_char_count[char]:
                    valid -= 1
    return ans




if __name__ == '__main__':
    print(smallestSubstringContaining("abcdefghijklmnopqrstuvwxyz","aajjttwwxxzz"))
