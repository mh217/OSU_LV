file = open('SMSSpamCollection.txt', encoding='utf8')
counth = 0
fullh =0 
counts = 0
fulls =0
exclamationcount =0

def count(wordss):
    global counth
    global counts
    global fullh
    global fulls
    if wordss[0] == 'ham' :
        counth +=1
        fullh +=(len(wordss)-1)
    else: 
        counts +=1
        fulls +=(len(wordss)-1)

def exclamation(wordss) :
    global exclamationcount
    if wordss[0] == 'spam' : 
        if wordss[len(wordss)-1].endswith('!') :
            exclamationcount +=1


for line in file : 
    line = line.rstrip()
    words = line.split()
    count(words)
    exclamation(words)
file.close()



avgh=fullh/counth
avgs=fulls/counts

print('Ham', avgh)
print('Spam', avgs)
print('Number of exclamations', exclamationcount)
