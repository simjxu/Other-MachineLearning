import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn import svm

with open('C:\\Users\\Simon\\Documents\\Data\\measurement_data_40_L3_z.json') as json_data:
    machine_data = json.load(json_data)

rng = np.random.RandomState(42)
measurements = machine_data['measurements']
measurements = sorted(measurements, key=lambda k: k['timestamp'])

X = []
rms = []
case = 2

for i in range(len(measurements)):
    fv = list()
    if case == 1:
        fv.append(measurements[i]['data']['z']['time_domain_features']['rms'])
        for fn in ['shaft_2x', 'shaft_1x', 'shaft_3x_to_10x_sum', 'shaft_3x']:
            fv.append(measurements[i]['data']['z']['frequency_domain_features'][fn])

        X.append(fv)

    if case == 2:
        fv = np.array(measurements[i]['data']['z']['frequency_domain']['amps'][0:400])
        X.append(fv)

    rms.append([measurements[i]['data']['z']['time_domain_features']['rms']])

X = np.array(X)
X_mean = X.mean(axis=0)
X -= X_mean
# fit the model
clf1 = IsolationForest(
    max_samples=100,
    random_state=rng
)
clf1.fit(X[0:300])
df1 = clf1.decision_function(X)
y_pred1 = clf1.predict(X)
an_count1 = sum(y_pred1 == -1)
print("Isolation Forest anomaly count: %s/%s" % (an_count1, y_pred1.size))

clf2 = svm.OneClassSVM(
    nu=0.01,
    kernel="rbf",
    gamma=0.1
)
clf2.fit(X[0:300])
df2 = clf2.decision_function(X)
y_pred2 = clf2.predict(X)
an_count2 = sum(y_pred2 == -1)
print("OCSVM anomaly count: %s/%s" % (an_count2, y_pred2.size))

plt.plot(rms)
plt.plot(df1)
plt.plot(y_pred1)
plt.title("Isolation Forest (Blue - RMS trend, Green - Anomaly score, Red - prediction(+1, -1))")
plt.show()
plt.clf()
plt.plot(rms)
plt.plot(df2)
plt.plot(y_pred2)
plt.title("One class SVM (Blue - RMS trend, Green - Hyperplane distance, Red - prediction(+1, -1))")
plt.show()