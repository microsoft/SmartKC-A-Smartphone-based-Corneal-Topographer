import os
import numpy as np
import matplotlib.pyplot as plt
file_name = 'kt_calib_gap2_config1.txt'

l = []
f = open(file_name)
for idx, line in enumerate(f):
	if(idx+1)%5 == 0:
		error = line.strip().split(' ')[-1]
		l.append(round(float(error), 2))
f.close()

print("Minimum error at: ", np.linspace(3, 6, 31)[np.argmin(l)], l[np.argmin(l)])

plt.figure()
plt.plot(np.linspace(3, 6, 31), l)
plt.xlabel('gap 2 in mm')
plt.ylabel('mean percentage error')
plt.title('Config1 gap2 tuning, gap1 is -2 mm')
plt.savefig('gap_2_tuning_config1.png')
plt.close()





exit()

#l.sort()
output = []
for idx in range(0, 11):
	output.append([])
for x in l[:]:
	error = x[0]
	gap1 = int(x[1].strip().split(' ')[3])
	gap2 = int(x[1].strip().split(' ')[7])
	output[gap2].append(round(error, 2))
	#print("Gap 1:", gap1, "Gap 2:", gap2, "Error", round(error, 2))

for idx in range(0, 11):
	plt.figure()
	plt.plot(range(-10, 11), output[idx])
	plt.xlabel('gap 1')
	plt.ylabel('error')
	plt.title('gap 2 is '+str(idx)+' mm')
	plt.savefig('plots_gap1/'+str(idx)+'.png')
	plt.close()
