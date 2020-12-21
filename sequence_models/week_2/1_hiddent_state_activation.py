import numpy as np


'''
Joining (Concatenation)
'''
w_hh = np.full((3, 2), 1) # returns an array of size 3x2 filled with all 1s
w_hx = np.full((3, 3), 9) # returns an array of size 3x3 filled with all 9s

# option1: concatenate horizontal
w_h1 = np.concatenate((w_hh, w_hx), axis=1)

# option2: hstack
w_h2 = np.hstack((w_hh, w_hx))


'''
Hidden State and Inputs
'''
h_t_prev = np.full((2, 1), 1) # returns an array of size 2x1 filled with all 1s
x_t = np.full((3, 1), 9) # returns an array of size 3x1 filled with all 9s

# option: concatenate vertical
ax_1 = np.concatenate((h_t_prev, x_t), axis=0)
ax_2 = np.vstack((h_t_prev, x_t))


'''
Verify formulas
'''
w_hh = np.full((3, 2), 1) # returns an array of size of 3x2 filled with all 1s
w_hx = np.full((3, 3), 9) # returns an array of size of 3x3 filled with all 9s
h_t_prev = np.full((2, 1), 1) # returns an array of size 2x1 filled with all 1s
x_t = np.full((3, 1), 9) # returns an array of size 3x1 filled with all 9s

stack_1 = np.hstack((w_hh, w_hx))
stack_2 = np.vstack((h_t_prev, x_t))

formula_1 = np.matmul(np.hstack((w_hh, w_hx)), np.vstack((h_t_prev, x_t)))
print(f"Formula1 output: {formula_1}")


formula_2 = np.matmul(w_hh, h_t_prev) + np.matmul(w_hx, x_t)
print(f"Formula2 output: {formula_2}")

print("-- Verify --")
print(f"Results are the same: {np.allclose(formula_1, formula_2)}")





















