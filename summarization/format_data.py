import csv
import string

new_file = open('Reviews.csv', 'w')
with open('cell.txt', 'r') as f:
	new_file.write("Summary	Text\n")
	for line in f:
		if "product/productId" in line:
			continue
		elif "product/title" in line:
			continue
		elif "product/price" in line:
			continue
		elif "product/userId" in line:
			continue
		elif "review/profileName" in line:
			continue
		elif "review/helpfulness" in line:
			continue
		elif "review/time" in line:
			continue
		elif "review/score" in line:
			continue
		elif "review/userId" in line:
			continue
		else:
			if line != "\n":
				if "review/summary" in line:
					new_file.write(line.strip('review/summary:').strip(" \n"))
					new_file.write("\t")
				else:
					line = string.replace(line, 'review/text: ', '')
					new_file.write(line.strip(" \n"))
			else:
				new_file.write(line)