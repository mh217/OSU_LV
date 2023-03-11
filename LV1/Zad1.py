
print('Koliko imate radnih sati?')
sati= int(input())
print('Placa po radnom satu?')
placa=float(input())
print('Broj radnih sati: ', sati,'h')
print('Placa:', placa, 'euro')

def total_euro(placauk) :
    print('Ukupno u eurima', + placauk,'euro')

total_euro(sati*placa)