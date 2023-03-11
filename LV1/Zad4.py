
def count(wordss) : 
    global word_dictionary
    count=0
    for word in wordss: 
        if word in word_dictionary:
            word_dictionary[word] +=1
        else:
            word_dictionary[word]= 1
            
def countOnes(word_dictionary):
    counter =0
    for key in list(word_dictionary.keys()) :
        if word_dictionary[key] == 1: 
            print(key,word_dictionary[key])
            counter += 1
    return counter

                

file = open('song.txt', 'r')
word_dictionary={}
for line in file: 
    line = line.rstrip()
    words = line.split()
    count(words)
    
    

print(word_dictionary)
print('Broj rijeci koje se pojavljuju samo jednom:', countOnes(word_dictionary))

