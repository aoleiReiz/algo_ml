def firstNonRepeatingCharacter(string):
    # Write your code here.
    counter = {}
    for c in string:
        if c not in counter:
            counter[c] = 1
        else:
            counter[c] += 1
    for idx, c in string:
        if counter[c] == 1:
            return idx
    return -1


def generateDocument(characters, document):
    # Write your code here.
    counter = {}
    for c in characters:
        if c not in counter:
            counter[c] = 1
        else:
            counter[c] += 1
    for d in document:
        if d not in counter:
            return False
        elif counter[d] == 0:
            return False
        else:
            counter[d] -= 1
    return True


def runLengthEncoding(string):
    n = len(string)
    if n > 0:
        ans = []
        cur_char = string[0]
        cur_count = 1
        for char in string[1:]:
            if cur_char == char:
                cur_count += 1
                if cur_count > 9:
                    ans.append(f"9{cur_char}")
                    cur_count -= 9
            else:
                ans.append(f"{cur_count}{cur_char}")
                cur_char = char
                cur_count = 1
        ans.append(f"{cur_count}{cur_char}")
        return "".join(ans)
    else:
        return ""


def caesarCipherEncryptor(string, key):
    # Write your code here.
    letters = "abcdefghijklmnopqrstuvwxyz"
    letter_2_idx = {c: idx for idx, c in enumerate(letters)}
    new_string_letters = [letters[(letter_2_idx[c] + key) % 26] for c in string]
    return "".join(new_string_letters)

