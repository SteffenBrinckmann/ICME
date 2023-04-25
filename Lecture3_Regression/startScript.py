# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#DOWNLOAD FROM https://github.com/SteffenBrinckmann/ICME
# OPTION 1
# - click on the file -> "RAW"
# - copy-paste into ...
#
# OPTION 2
# - click on green "Code" and download zip-file
dataset= pd.read_csv('ScratchArea.csv') 

#visually check data, first
print(dataset.head())

plt.plot(dataset['Material'], dataset['scratch area 1'],'o')
plt.xlabel('Load in $mN$')
plt.ylabel('Scratch Area in $\mu m^2$')
plt.show()

#sns.displot(dataset['scratch area 1'], bins=20, kde=True)
#plt.show()
sns.countplot(x="Material", hue="Tip", data=dataset)
plt.show()
sns.pairplot(dataset, hue='Material')
plt.show()
