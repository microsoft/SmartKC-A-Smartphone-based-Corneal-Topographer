import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score


def my_plot(x, y, name="temp.png"):
	plt.figure()
	plt.plot(range(1, len(x)+1), x, label='tilt')
	plt.plot(range(1, len(y)+1), y, label='non-tilt')
	plt.xlabel('ids ->')
	plt.ylabel('curvature_diff')
	plt.legend()
	plt.savefig(name)
	plt.close()

file_name = '../data/dr_survey_responses.csv'
labels_dict = {}
f = open(file_name)
for line in f:
	line = line.strip().split(',')
	labels_dict[line[0]] = int(line[8])
f.close()

file_name = '../tilt_scores_fine_tuned.txt'
tilt_dict = {}
f = open(file_name)
for line in f:
	if "max_steep" in line:
		line = line.strip().split(' ')
		tilt_score = round(337.5*(float(line[2])-float(line[4])), 2)
		name = line[0].strip().split('_')
		name = name[0]+"_"+name[1]
		if name not in tilt_dict:
			tilt_dict[name] = []
		tilt_dict[name].append(tilt_score)
f.close()

x, y = [], []

for name in tilt_dict:
	if name not in labels_dict:
		continue
	x.append(tilt_dict[name])
	y.append(labels_dict[name])

x = np.array(x)
y = np.array(y)

#print(x.shape)
print("Tilt", x[y==1][:,0].mean(), x[y==1][:,1].mean())
print("Non-Tilt", x[y==0][:,0].mean(), x[y==0][:,1].mean())

my_plot(x[y==1][:,0], x[y==0][:,0], name="tan_tilt.png")
my_plot(x[y==1][:,1], x[y==0][:,1], name="axial_tilt.png")

plt.figure()
for idx in range(len(y)):
	if y[idx] == 1:
		plt.scatter(x[idx][0], x[idx][1], color="red", s=1, marker='o', linewidths=2)
	else:
		plt.scatter(x[idx][0], x[idx][1], color="blue", s=1, marker='o', linewidths=2)
plt.xlabel('tan_tilt')
plt.ylabel('axial_tilt')
plt.title('scatter plot for tilts')
plt.savefig('tilt_scatter.png')
plt.close()

X = x.copy()
from mlxtend.plotting import plot_decision_regions
svm = SVC(kernel='linear')
svm.fit(X, y)
plot_decision_regions(X, y, clf=svm, legend=2)
plt.savefig('decision_boundary.png')
plt.close()


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

import random
random.seed(42)
idx = list(range(0,len(X)))
random.shuffle(idx)
idx = np.array(idx)

X = x.copy()
model = SVC(C=0.5, kernel='linear')
clf = model.fit(X[idx[:20]], y[idx[:20]])

preds = clf.predict(X[idx[20:]])
print("Accuracy", accuracy_score(y[idx[20:]], preds))

fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface of linear SVC ')
# Set-up grid for plotting.
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_xlabel('tan_tilt')
ax.set_ylabel('axial_tilt')
#ax.set_xticks(())
#ax.set_yticks(())
ax.set_title(title)
ax.legend()
plt.savefig('1_tilt_decision.png')
