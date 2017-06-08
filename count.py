import csv

f = open('sample_a.csv', 'r')

reader = csv.reader(f)
for row in reader:
    print row

f.close()

element_counts = {}

for element in data:#data変数の定義必要
    if element_counts.has_key(element):
        element_counts[element] += 1
    else:
        element_counts[element] = 1

print element_counts
