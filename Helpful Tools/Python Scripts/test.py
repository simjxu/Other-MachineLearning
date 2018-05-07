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

# Remove the bad ones from X
acc_832m1_x[167, :] = avg_x
acc_832m1_x[100, :] = avg_x
acc_832m1_x[135, :] = avg_x
acc_832m1_x[232, :] = avg_x
acc_832m1_x[10, :] = avg_x
acc_832m1_x[215, :] = avg_x

# Remove the bad ones from Y
acc_832m1_y[26, :] = avg_y
acc_832m1_y[167, :] = avg_y
acc_832m1_y[232, :] = avg_y
acc_832m1_y[327, :] = avg_y
acc_832m1_y[135, :] = avg_y
acc_832m1_y[29, :] = avg_y


# Remove the bad ones from Z
acc_832m1_z[232, :] = avg_z
acc_832m1_z[167, :] = avg_z
acc_832m1_z[100, :] = avg_z
acc_832m1_z[183, :] = avg_z
acc_832m1_z[135, :] = avg_z
acc_832m1_z[145, :] = avg_z
acc_832m1_z[76, :] = avg_z
acc_832m1_z[31, :] = avg_z
acc_832m1_z[10, :] = avg_z
acc_832m1_z[179, :] = avg_z


for i in range(num_meas):
    argmax_x[i] = np.argmax(acc_832m1_x[:, i])
    argmax_y[i] = np.argmax(acc_832m1_y[:, i])
    argmax_z[i] = np.argmax(acc_832m1_z[:, i])
    argmin_x[i] = np.argmin(acc_832m1_x[:, i])
    argmin_y[i] = np.argmin(acc_832m1_y[:, i])
    argmin_z[i] = np.argmin(acc_832m1_z[:, i])


print(argmax_x)
print(argmax_y)
print(argmax_z)

# print(argmin_x)
# print(argmin_y)
# print(argmin_z)

# print(max_x)
# print(max_y)
# print(max_z)
# print(min_x)
# print(min_y)
# print(min_z)
# print(avg_x)
# print(avg_y)
print(avg_z)
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


transposed = np.transpose(acc_832m1_z)

plt.plot(transposed)
plt.show()
