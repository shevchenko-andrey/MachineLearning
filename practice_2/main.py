from numbers import Number
import numpy as np

vector = np.array([1,2,3])
print(f"np.array vector: {type(vector)}\n{vector}")
matrix = np.array([[1,2,3], [4,5,6], [7,8,9]])
print(f"np.array matrix:\n{matrix}")
zero_matrix = np.zeros((4,5), dtype=Number)
print(f"np.zeros zero_matrix 4*5:\n{zero_matrix}")
identity_matrix = np.identity(3)
print(f"np.identity 3*3:\n{identity_matrix}")
print(f"matrix size: {matrix.size}")
matrix_a = np.array([[1,2,3], [4,5,6]])
matrix_b = np.array([[-1,-2,-3], [-4,-5,-6]])
print(f"matrix_a + matrix_b =\n{matrix_a + matrix_b}")
print(f"matrix_a - matrix_b =\n{matrix_a - matrix_b}")
print(f"matrix_a * matrix_b =\n{matrix_a * matrix_b}")
b = 2
print(f"matrix_a + b =\n{matrix_a + b}")
matrix_c = np.array([[1,2,3], [4,5,6]])
matrix_d = np.array([[1,1], [1,1], [1,1]])
print(f"matrix_c @ matrix_d =\n {matrix_c @ matrix_d}")
vector_v = np.array([1,2,3])
vector_u = np.array([1,1,1])

def calc_euclidean_norm(vector_to_calc: [Number]):
    sum_of_squares = 0
    for component in vector_to_calc:
        sum_of_squares += component ** 2
    return np.sqrt(sum_of_squares)

print(f"Euclidean norm custom realization: ||vector_v|| = {calc_euclidean_norm(vector_v)}")
print(f"Euclidean norm custom realization: ||vector_u|| = {calc_euclidean_norm(vector_u)}")
print(f"np.linalg.norm: ||vector_v|| = {np.linalg.norm(vector_v)}")
print(f"np.linalg.norm: ||vector_u|| = {np.linalg.norm(vector_u)}")
vector_g = [11,22,33,44,55]
print(f"vector_g[-2:] = {vector_g[-2:]}")
