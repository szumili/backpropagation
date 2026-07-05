def get_matrix(height, width, grid):

    matrix = []

    for row in range(height):
        rows = []
        for col in range(width):
            rows.append(grid[row][col])
        matrix.append(rows)

    return matrix