
def funk(br) :
    if br >= 0.9: 
        print('A')
    elif br>=0.8: 
        print('B')
    elif br>=0.7: 
        print('C')
    elif br>=0.6: 
        print('D')
    elif br< 0.6: 
        print('F')


def main() :
    print('Unesite broj')

    while True : 
        try :
            number = float(input())
        except: 
            print('Put in a number')
            main()
        if number > 0.0 and number < 1.0 : 
            funk(number)
        else:
            print('Prešli ste odgovarajuci interval')
            main()
    
main()   


