import os

names = []
directory = "./datasets/"

for i in os.listdir(directory + 'music'):
    names.append(i)
    # print(i)

path_wave = directory + 'wave/'
if not os.path.exists(path_wave):
    os.makedirs(path_wave)

data_np = []
i = 0

#convert music in wav
for name in names:
    print(name)
    fname = directory + 'music/' + name
    oname = path_wave + str(i) + '.wav'
    cmd = 'lame --decode {0} {1}'.format(fname, oname)
    os.system(cmd)
    i += 1

