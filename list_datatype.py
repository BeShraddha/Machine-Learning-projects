'''

Welcome to GDB Online.
GDB online is an online compiler and debugger tool for C, C++, Python, Java, PHP, Ruby, Perl,
C#, VB, Swift, Pascal, Fortran, Haskell, Objective-C, Assembly, HTML, CSS, JS, SQLite, Prolog.
Code, Compile, Run and Debug online from anywhere in world.

'''
list_no, eq, value = input().split()
list_no = value[1:-1].split(',')
l = []
for i in list_no:
    if int(i)>= 0:
        l.append(int(i))
print(l)