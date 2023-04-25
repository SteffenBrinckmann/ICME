# -*- coding: utf-8 -*-
###        0          1        2
rooms = ['AH1', 'corridor', 'AH2', 'roof']
roomNumber = 1
while True:
    answer = input('\nWhere to go? ')
    if answer.strip()=='up':
        roomNumber = roomNumber + 1
    elif answer.strip()=='down':    
        roomNumber = roomNumber - 1
    else:
        print("I did not understand you.")
    if rooms[roomNumber]=='roof':
        break
    if rooms[roomNumber]=='AH1':
        answer = input('What is the most important python thing? ')
        if answer == 'tab':
            print('Great you learned something')
        else:
            print('ERROR you did not listen!!! I send you to the lecture.')
            roomNumber = 2
    print('You are now in room:', rooms[roomNumber])
print()
print('Welcome!!: you have left the house of python horror.')
