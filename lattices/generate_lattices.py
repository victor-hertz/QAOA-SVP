import numpy as np
import sys

def ideal_mat(params):
  d = len(params)
  a = []
  for i in range(d):
    a.append(np.roll(params,i))
  return np.array(a)

def generate_mat(d, limit, cyclic):
  i = True
  while(i):
    if cyclic:
      a = ideal_mat(np.random.randint(-limit,limit+1, size=d))
    else:
      a = np.random.randint(-limit,limit+1, size=(d,d))
    i = np.isclose(np.linalg.det(a), 0)
    #i=0
  return a

def check_uni(a, b):
  x = np.matmul(np.linalg.inv(b) ,a)
  if not np.isclose(np.round(x),x).all():
    return False
  if not np.isclose(np.abs(np.linalg.det(x)),1):
    return False
  return True

# Parameters
unimod = True
cyclic = True
sample_s = 1000
d = 6
limit = 10
save= True

path = cwd = sys.path[0]
mat_list = []
print("d:", d)
i = 0

while i < sample_s:
  if i % int(sample_s * 0.1) == 0 and i > 0:
    print("Percentage:", (i + 1) / sample_s * 100)
  a = generate_mat(d, limit, cyclic)
  if unimod:
    m_pass=True
    for j in mat_list:
      if check_uni(j, a):
        print(j,a)
        m_pass=False
        break
    if m_pass:
      mat_list.append(a)
      i+=1
  else:
    if not any((a == mat).all() for mat in mat_list):
      mat_list.append(a)
      i += 1
mat_list = np.array(mat_list)

if cyclic:
  type='cyclic'
else:
  type='generic'

lattice_type = type+'_'+str(d)+'d_'+str(limit)+'r_'+str(int(sample_s/1000))+'k'
if unimod:
  lattice_type+='_un'
print(lattice_type)

if save:
  with open(path+'/mat_'+lattice_type+'.npy', 'wb') as f:
      np.save(f, mat_list)
