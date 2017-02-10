import os.path

def get_name(n, case):
	prefix = "../dataset/dataset3.3a/set"
	suffix = ""
	
	if case == 1:
		suffix = "/temp.txt"
	elif case == 2:
		suffix = "/HDR_Classlabel.txt"
	elif case == 3:
		suffix = "/HDR_TRIG.txt"

	fname = prefix+str(n)+suffix
	return fname

fname_target = get_name(3, 2)
fname_artifact = "../dataset/dataset3.3a/set3/HDR_ArtifactSelection.txt"
counter = 0

try:
	with open(fname_target) as target_file, open(fname_artifact) as artifact_file:
		while True:
			target_line = target_file.readline()
			artifact_line = artifact_file.readline()

			if artifact_line == "":
				break

			if int(artifact_line) == 0 and not target_line == "NaN\n":
				counter+=1	

		print counter

except IOError as e:
	print 'Operation failed: %s' % e.strerror