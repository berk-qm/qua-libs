import numpy as np


N = np.random.randint(0, 11520)
case = 1 if (576 <= N <5670) else 0
case = 2 if (5670 <= N < 10944) else case
case = 3 if (N > 10944) else case
is_case_12 = 9 if (576 <= N < 10944) else 1
if case == 1:
    N -= 576
elif case == 2:
    N -= 5670
elif case==3:
    N -= 10944


c1 = N // is_case_12 // 24
c2 = N // is_case_12 % 24
s1 = (N//3) % 3
s2 = N % 3

print(N)
print(c1, c2, s1, s2)
print("paulis")
print(c1 % 4, c1 // 4)
print(c2 % 4, c2 // 4)

if case ==0:
    un= np.unravel_index(N, (24,24,))
elif case ==1:
    un = np.unravel_index(N, (24,24,3,3))
elif case ==2:
    un = np.unravel_index(N, (24,24,3,3))
elif case ==3:
    un = np.unravel_index(N, (24, 24))

print(un)
print(f"pauli1 = {np.unravel_index(un[0], (6,4,))}, pauli2= {np.unravel_index(un[1], (6,4,))}")
