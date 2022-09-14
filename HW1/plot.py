from matplotlib import pyplot as plt

log_x = [-1, -2, -3, -4]
log_fd_error = [ 0.2399 , -0.6705 , -1.6612 , -2.6603]
log_cd_error = [ -0.2774 , -2.2850 , -4.2851 , -6.2850]
plt.scatter(log_x, log_fd_error,c="r")
plt.scatter(log_x, log_cd_error,c="b")
plt.legend(['forward differences','central differences'])
plt.savefig('plot.png')
