# -*- coding: utf-8 -*-


# 1. Import libraries

# 2. Initialize stuff, read in data

# 3. Use, calculate something, do something new

# 4. Show, print do,

"""
# List
#shopping = list('eggs', 'milk')
shopping = ['eggs', 'milk', 42, 'milk']
print("List",shopping)

element = shopping.pop(0)
element += "ssss"
print("first",element)
shopping.append('bread')
print('Remainder',shopping)
"""

shopping = set(['eggs', 'milk', 42])
"""
print("Set",shopping)
print('any',shopping.pop())
shopping.add('milk')
print('Set later in time', shopping)
"""

# dictionary: key-value pairs
engl2german = {"house":"Haus", "car": "Auto"}
for key in engl2german.keys():
    print(key, engl2german[key])
print(  engl2german['car'] )

response = {'status':404, 'content': shopping}



