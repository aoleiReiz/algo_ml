def longestBalancedSubstring(string):
    max_len = 0
    for i in range(len(string)):
        for j in range(i + 2, len(string) + 1, 2):
            if isBalanced(string[i:j]):
                max_len = max(max_len, j - i)
    return max_len


def isBalanced(string):
    open_parens_stack = []
    for char in string:
        if char == "(":
            open_parens_stack.append(char)
        elif len(open_parens_stack) > 0:
            open_parens_stack.pop()
        else:
            return False
    return len(open_parens_stack) == 0


def longestBalancedSubstring2(string):
    ans = 0
    idx_stack = [-1]
    for i in range(len(string)):
        if string[i] == "(":
            idx_stack.append(i)
        else:
            idx_stack.pop()
            if len(idx_stack) == 0:
                idx_stack.append(i)
            else:
                ans = max(ans, i - idx_stack[-1])
    return ans


def longestBalancedSubstring3(string):
    ans = 0
    opening_count = 0
    closing_count = 0
    for c in string:
        if c == "(":
            opening_count += 1
        else:
            closing_count += 1
        if closing_count > opening_count:
            opening_count = 0
            closing_count = 0
        elif closing_count == opening_count:
            ans = max(ans, opening_count * 2)
    opening_count = 0
    closing_count = 0
    for c in string[::-1]:
        if c == "(":
            opening_count += 1
        else:
            closing_count += 1
        if opening_count > closing_count:
            opening_count = 0
            closing_count = 0
        elif closing_count == opening_count:
            ans = max(ans, opening_count * 2)

    return ans




if __name__ == '__main__':
    print(longestBalancedSubstring3("(()))("))