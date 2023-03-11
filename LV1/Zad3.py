
def funk(listofnumbers) :
    print(listofnumbers)
    size=len(listofnumbers)
    if size == 0 :
        print('Lista je prazna')
    else:
        print(size)
        zbroj = sum(listofnumbers)
        average=zbroj/int(size)
        print(average)
        print(min(listofnumbers))
        print(max(listofnumbers))
        sorted_list = sorted(listofnumbers)
        print(sorted_list)
    

def loop() :
    numberlist = []
    while True:
        number = input('Unesite broj:')
        if number == "Done":
            print('This is the end')
            funk(numberlist)
            break
        elif not number.isdigit():
            print('Unesite broj!')
        else:
            numberlist.append(float(number))


loop()