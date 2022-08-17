import argparse

#I also implemented using sys library (uncomment the below codes):
'''
import sys
a = sys.argv[1:]
if a[-2]=="--p":
    p = int(a[-1])
    a = a[0:-2]
    v = list(map(float, a))

else:
    v = list(map(float,a))
    p = 2.0
'''
# Find the p-norm of a given list:
parser = argparse.ArgumentParser()
parser.add_argument('v', type = float, nargs = '+')
parser.add_argument('--p', type = int)
parse = parser.parse_args()
summ = 0;
if parse.p:
    p = parse.p
else:
    p=2
for i in parse.v:
    summ += abs(i**p)
print("Norm of {} is {:.2f}".format(parse.v,summ**(1/p)))
