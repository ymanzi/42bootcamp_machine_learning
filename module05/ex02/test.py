#!/usr/bin/python

from TinyStatistician import TinyStatistician
tstat = TinyStatistician()

a = [1, 42, 300, 10, 59]

print(tstat.mean(a))
print(tstat.median(a))
print(tstat.quartiles(a, 25))
print(tstat.quartiles(a, 75))
print(tstat.var(a))
print(tstat.std(a))
