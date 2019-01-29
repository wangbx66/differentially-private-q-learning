import sys

sigmas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

for sigma in sigmas:
    for seed in range(10):
        print(sigma, seed)
        sys.argv = ['x', str(seed), str(sigma)]
        exec(open('dpql.py').read())
