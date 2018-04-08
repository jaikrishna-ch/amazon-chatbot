import pandas as pd
import gzip
import json
import numpy as np
with open('./data.json', 'r') as f:
    data_dict = json.load(f)

print "Read the product json..."

asin_set = set()
for asin in data_dict:
	asin_set.add(asin)




def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

# df = getDF('/home/jaikrishna/Desktop/btp/spell-match/reviews_Cell_Phones_and_Accessories.json.gz')
df = pd.read_json('./reviews_Cell_Phones_and_Accessories.json', lines=True)
print "Read the reviews dataframe..."
df1 =  df[['asin', 'reviewText', 'overall', 'summary']]	
count = 0
asin = []
reviewText = []
overall = []
summary = []
for index, row in df1.iterrows():
	count += 1
	if(count % 10000 == 0):
		print "processed " + str(count) + " rows"
	try:
		if row['asin'] not in asin_set:
			continue
		asin.append(row['asin'])
	except:
		asin.append("NONE")
	try:
		reviewText.append(row['reviewText'].lower())
	except:
		reviewText.append("reviewText")
	try:
		summary.append(row['summary'].lower())
	except:
		summary.append("summary")
	try:
		overall.append(row['overall'])
	except:
		overall.append("overall")


asin_linkup = {}
max_num_reviews = 5

# For the sake of demo, inorder to have a faster load time we shall consider only a fixed number of reviews per asin.

print "Total number of asins " + str(np.unique(asin).size)
print "Total number of reviews " + str(len(asin))
num_asins = 0

for i in range(0, len(asin)):  # this is the number of different reviews.
	if asin[i] not in asin_linkup:
		if num_asins == 50000:
			break																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																	
		num_asins += 1
		if(num_asins % 100 == 0):
			print str(num_asins) + " asins completed ! "
		asin_linkup[asin[i]] = {}
		count = 1
		asin_linkup[asin[i]][(count)] = [reviewText[i], overall[i], summary[i]]
	else:
		if count == max_num_reviews + 1:
			continue
		count += 1
		asin_linkup[asin[i]][(count)] = [reviewText[i], overall[i], summary[i]]


with open('reviews.json', 'w') as fp:	
	json.dump(asin_linkup, fp, sort_keys=True, indent=4)
