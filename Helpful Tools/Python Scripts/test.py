import numpy as np
import csv
import matplotlib.pyplot as plt

num_meas = 15

reader = csv.reader(open("data/acc_lsm_x.csv", "r"), delimiter=",")
x = list(reader)
acc_832m1_x = np.array(x).astype("float")

reader = csv.reader(open("data/acc_lsm_y.csv", "r"), delimiter=",")
x = list(reader)
acc_832m1_y = np.array(x).astype("float")

reader = csv.reader(open("data/acc_lsm_z.csv", "r"), delimiter=",")
x = list(reader)
acc_832m1_z = np.array(x).astype("float")

avg_x = np.zeros(num_meas); avg_y = np.zeros(num_meas); avg_z = np.zeros(num_meas)
med_x = np.zeros(num_meas); med_y = np.zeros(num_meas); med_z = np.zeros(num_meas)
min_x = np.zeros(num_meas); min_y = np.zeros(num_meas); min_z = np.zeros(num_meas)
max_x = np.zeros(num_meas); max_y = np.zeros(num_meas); max_z = np.zeros(num_meas)
argmax_x = np.zeros(num_meas); argmax_y = np.zeros(num_meas); argmax_z = np.zeros(num_meas)
argmin_x = np.zeros(num_meas); argmin_y = np.zeros(num_meas); argmin_z = np.zeros(num_meas)




# Remove the bad ones from X
acc_832m1_x[152, :] = np.ones((1,num_meas)) *0.01   # Z
acc_832m1_x[107, :] = np.ones((1,num_meas))*0.01      # Z
acc_832m1_x[113, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_x[235, :] = np.ones((1,num_meas))*0.01      # Z
acc_832m1_x[236, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_x[269, :] = np.ones((1,num_meas))*0.01      # Z
acc_832m1_x[270, :] = np.ones((1,num_meas))*0.01      # Z
acc_832m1_x[55, :] = np.ones((1,num_meas))*0.01      # Z
acc_832m1_x[99, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_x[234, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_x[117, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_x[81, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_x[54, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_x[129, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_x[109, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_x[50, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_x[148, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_x[78, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_x[202, :] = np.ones((1,num_meas)) *0.01     # X
acc_832m1_x[251, :] = np.ones((1,num_meas)) *0.01     # X
acc_832m1_x[241, :] = np.ones((1,num_meas)) *0.01     # X
acc_832m1_x[131, :] = np.ones((1,num_meas)) *0.01     # X

acc_832m1_x[259, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_x[106, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_x[118, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_x[101, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_x[72, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_x[95, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_x[149, :] = np.ones((1,num_meas)) *0.01     # Z

acc_832m1_x[150, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_x[63, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_x[188, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_x[48, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_x[196, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_x[146, :] = np.ones((1,num_meas)) *0.01     # Z

# Remove the bad ones from Y
acc_832m1_y[152, :] = np.ones((1,num_meas)) *0.01   # Z
acc_832m1_y[107, :] = np.ones((1,num_meas))*0.01      # Z
acc_832m1_y[113, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_y[235, :] = np.ones((1,num_meas))*0.01      # Z
acc_832m1_y[236, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_y[269, :] = np.ones((1,num_meas))*0.01      # Z
acc_832m1_y[270, :] = np.ones((1,num_meas))*0.01      # Z
acc_832m1_y[55, :] = np.ones((1,num_meas))*0.01      # Z
acc_832m1_y[99, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_y[234, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_y[117, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_y[81, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_y[54, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_y[129, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_y[109, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_y[50, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_y[148, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_y[78, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_y[202, :] = np.ones((1,num_meas)) *0.01     # X
acc_832m1_y[251, :] = np.ones((1,num_meas)) *0.01     # X
acc_832m1_y[241, :] = np.ones((1,num_meas)) *0.01     # X
acc_832m1_y[131, :] = np.ones((1,num_meas)) *0.01     # X

# Remove the bad ones from Z
acc_832m1_z[152, :] = np.ones((1,num_meas)) *0.01   # Z
acc_832m1_z[107, :] = np.ones((1,num_meas))*0.01      # Z
acc_832m1_z[113, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_z[235, :] = np.ones((1,num_meas))*0.01      # Z
acc_832m1_z[236, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_z[269, :] = np.ones((1,num_meas))*0.01      # Z
acc_832m1_z[270, :] = np.ones((1,num_meas))*0.01      # Z
acc_832m1_z[55, :] = np.ones((1,num_meas))*0.01      # Z
acc_832m1_z[99, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_z[234, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_z[117, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_z[81, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_z[54, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_z[129, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_z[109, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_z[50, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_z[148, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_z[78, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_z[202, :] = np.ones((1,num_meas)) *0.01     # X
acc_832m1_z[251, :] = np.ones((1,num_meas)) *0.01     # X
acc_832m1_z[241, :] = np.ones((1,num_meas)) *0.01     # X
acc_832m1_z[131, :] = np.ones((1,num_meas)) *0.01     # X


acc_832m1_z[259, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_z[106, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_z[118, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_z[101, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_z[72, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_z[95, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_z[149, :] = np.ones((1,num_meas)) *0.01     # Z

acc_832m1_z[150, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_z[63, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_z[188, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_z[48, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_z[196, :] = np.ones((1,num_meas)) *0.01     # Z
acc_832m1_z[146, :] = np.ones((1,num_meas)) *0.01     # Z


for i in range(num_meas):
    avg_x[i] = np.average(acc_832m1_x[:, i])
    avg_y[i] = np.average(acc_832m1_y[:, i])
    avg_z[i] = np.average(acc_832m1_z[:, i])
    med_x[i] = np.median(acc_832m1_x[:, i])
    med_y[i] = np.median(acc_832m1_y[:, i])
    med_z[i] = np.median(acc_832m1_z[:, i])
    min_x[i] = np.min(acc_832m1_x[:, i])
    min_y[i] = np.min(acc_832m1_y[:, i])
    min_z[i] = np.min(acc_832m1_z[:, i])
    max_x[i] = np.max(acc_832m1_x[:, i])
    max_y[i] = np.max(acc_832m1_y[:, i])
    max_z[i] = np.max(acc_832m1_z[:, i])
    argmax_x[i] = np.argmax(acc_832m1_x[:, i])
    argmax_y[i] = np.argmax(acc_832m1_y[:, i])
    argmax_z[i] = np.argmax(acc_832m1_z[:, i])
    argmin_x[i] = np.argmin(acc_832m1_x[:, i])
    argmin_y[i] = np.argmin(acc_832m1_y[:, i])
    argmin_z[i] = np.argmin(acc_832m1_z[:, i])


print(argmax_x)
print(argmax_y)
print(argmax_z)

# print(max_x)
# print(max_y)
# print(max_z)
# print(min_x)
# print(min_y)
# print(min_z)
# print(avg_x)
# print(avg_y)
# print(avg_z)
# print(med_x)
# print(med_y)
# print(med_z)

# plt.plot(avg_x, 'b')
# plt.plot(med_x, 'k')
# plt.plot(min_x, 'r')
# plt.plot(max_x, 'g')
# plt.show()
# 
# plt.plot(avg_y, 'b')
# plt.plot(med_y, 'k')
# plt.plot(min_y, 'r')
# plt.plot(max_y, 'g')
# plt.show()
# 
# plt.plot(avg_z, 'b')
# plt.plot(med_z, 'k')
# plt.plot(min_z, 'r')
# plt.plot(max_z, 'g')
# plt.show()


transposed = np.transpose(acc_832m1_y)

plt.plot(transposed)
plt.show()
