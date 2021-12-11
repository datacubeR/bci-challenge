from box import Box

dict = {'hola': 1, 'chao': 2}

dict = Box(dict)

print(dict.hola)
print(dict.chao)