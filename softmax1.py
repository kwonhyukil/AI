import numpy as np

one_hot = np.eye(4)

print(one_hot,"\n\n")

y_list = [0, 1, 0, 3, 2, 3]
one_hot_value = (one_hot[y_list])

print(one_hot_value)

print(np.argmax(one_hot_value, axis=1))


x = np.array([[2,1,2],
              [2,3,4],
              [5,6,7]])

print(x.argmax())