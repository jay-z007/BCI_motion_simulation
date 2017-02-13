import os.path

root_dir = os.path.abspath('./')
data_dir = os.path.join(root_dir, '../dataset', 'dataset3.5','data_psd')

files = [
'train_subject1_psd01.asc', 'train_subject1_psd02.asc', 'train_subject1_psd03.asc',
'train_subject2_psd01.asc', 'train_subject2_psd02.asc', 'train_subject2_psd03.asc',
'train_subject3_psd01.asc', 'train_subject3_psd02.asc', 'train_subject3_psd03.asc'
]

files_new = [
'train_subject1_psd1.asc', 'train_subject1_psd2.asc', 'train_subject1_psd3.asc',
'train_subject2_psd1.asc', 'train_subject2_psd2.asc', 'train_subject2_psd3.asc',
'train_subject3_psd1.asc', 'train_subject3_psd2.asc', 'train_subject3_psd3.asc'
]
for file in range(len(files)):
	file_name = os.path.join(data_dir, files[file])
	with open(file_name) as f, open(os.path.join(data_dir, files_new[file]), "w") as f1:
	    lines = f.readlines()
	    for line in lines:
		    arr = line.split()
	            if arr[-1] == "7.0000000e+00":
	                continue
		    f1.write(line)