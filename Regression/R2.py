import numpy as np
import math


def compute_correlation(X, Y):
    x_bar = np.mean(X)
    y_bar = np.mean(Y)
    SSR = 0
    var_X = 0
    var_Y = 0
    for i in range(0, len(X)):
        diff_xx_bar = X[i] - x_bar
        diff_yy_bar = Y[i] - y_bar
        SSR += diff_xx_bar*diff_yy_bar
        var_X += diff_xx_bar**2
        var_Y += diff_yy_bar**2

        SST = math.sqrt(var_X*var_Y)
        return SSR/SST

test_x = [1, 3, 8, 7, 9]
test_y = [10, 12, 24, 21, 34]
R_ret = compute_correlation(test_x, test_y)
print(R_ret)


    
