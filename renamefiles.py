import os

path = 'I:\\ML\\Projs\\videoclass\\bowl'
files = os.listdir(path)

for index, file in enumerate(files):
    os.rename(os.path.join(path, file), os.path.join(path, 'train{}.jpg'.format(index) ))

path2 = 'I:\\ML\\Projs\\videoclass\\bowl2'
files1 = os.listdir(path2)
index = index + 1

for i,file in enumerate(files1):
	os.rename(os.path.join(path2, file), os.path.join(path2, 'train{}.jpg'.format(index + i)))
