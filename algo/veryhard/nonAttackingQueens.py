def nonAttackingQueens(n):
    # Write your code here.
    column_replacements = [0] * n
    return get_number_of_non_attacking_replacements(0, column_replacements, n)


def get_number_of_non_attacking_replacements(row, column_placements, board_size):
    if row == board_size:
        return 1

    valid_placements = 0
    for col in range(board_size):
        if is_non_attacking_replacement(row, col, column_placements):
            column_placements[row] = col
            valid_placements += get_number_of_non_attacking_replacements(row + 1, column_placements, board_size)
    return valid_placements


def is_non_attacking_replacement(row, col, column_placements):
    for prev_row in range(row):
        column_to_check = column_placements[prev_row]
        same_column = column_to_check == col
        on_diagnose = abs(column_to_check - col) == row - prev_row
        if same_column or on_diagnose:
            return False
    return True


if __name__ == '__main__':
    print(nonAttackingQueens(4))