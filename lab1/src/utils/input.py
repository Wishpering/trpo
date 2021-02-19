def parse_matrix(stdin_input: str) -> list:
    matrix = []

    for array in stdin_input.split(';'):
        tmp = []

        for element in array.split(','):
            if element:
                tmp.append(float(element))

        if len(tmp) > 0:
            matrix.append(tmp)

    return matrix
