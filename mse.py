def mse(yt, yp):
    return 0.5 * (yt - yp) ** 2

def mse_prime(yt, yp):
    return yp - yt