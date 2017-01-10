import os.path

file_values = []
data = []
target = []

fname = "./dataset/subject1_ascii/train_subject1_raw01.asc"

if os.path.exists(fname):
	with open(fname) as file:
		while True:
			line = file.readline()
			if line == "":
				break
			arr = line.split()
			if arr[-1] == "7.0000000e+000":
				#print '7\n'
				continue
			for i in range(len(arr)-1):
				arr[i] = float(arr[i])/1
			
			data.append(arr[0:32])
			target.append(float(arr[-1]))
else:
	print "wrong"
#avg, min-max, SD, median, energy, 