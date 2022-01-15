import pandas as pd
import numpy as np
import math
import ipywidgets as widgets

import matplotlib.pyplot as plt
import seaborn as sns
import scipy
plt.rcParams["figure.figsize"] = (8,8)

class edaDF:
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.cat = []
        self.num = []

    def info(self):
        return self.data.info()
#DATA PREP
    def giveTarget(self):
        return self.target
        
    def setCat(self, catList):
        self.cat = catList
    
    def setNum(self, numList):
        self.num = numList
    
    def cat_to_numeric(self, data, columnName, current_val, desired_val):
        data[columnName].replace({current_val: desired_val}, inplace=True)
        return self.data.head()
#NUMERIC
    def numericstats(self, data):
        d = print('Nulls:\n\n\n', self.data.isna().sum())
        val = self.data.value_counts().T
        sort = print('Value Counts:\n\n\n',val.sort_values(ascending=False))
        desc = print('Full description:\n\n\n ', self.data.describe(include="all"))
        return d, sort, desc

    def describe(self, data):
        return self.data.describe()

    def quicklook(self, data):
        head = print('Head:\n', self.data.head())
        tail = print('Tail:\n', self.data.tail())
        val = data.value_counts().T
        sort = print('Value Counts:\n',val.sort_values(ascending=False))
        desc = print('Full description:\n ', df.describe(include="all"))
        return head, tail, sort, desc

#CORRELATION  
    def pair(self, data):
        plot = sns.pairplot(self.data, hue=None)
        plt.show()
        return plot
        
    def heatmap(self, data):
        df2 = self.data.apply(pd.to_numeric, errors='coerce')
        df2 = self.data.corr()
        plt.rcParams["figure.figsize"] = (20,10)
        h = sns.heatmap(df)
        plt.show()
        return h 

    def hists(self, data, show=True):
        n = len(self.data)
        cols = 2
        figure, ax = plt.subplots(math.ceil(n/cols), cols)
        plt.rcParams["figure.figsize"] = (20,10)
        figure = self.data.hist()
        if show == True:
            figure.show()
        return figure
#FULL EDA
    def fullEDA(self):
        out1 = widgets.Output()
        out2 = widgets.Output()
        out3 = widgets.Output()
        out4 = widgets.Output()

        tab = widgets.Tab(children = [out1, out2, out3, out4])
        tab.set_title(0, 'Info')
        tab.set_title(1, 'Numeric Stats')
        tab.set_title(2, 'Correlations')
        tab.set_title(3, 'Hists')
        display(tab)

        with out1:
            self.data.info()

        with out2:
            d = print('Nulls:\n\n\n', self.data.isna().sum())
            val = self.data.value_counts().T
            sort = print('Value Counts:\n\n\n',val.sort_values(ascending=False))
            desc = print('Full description:\n\n\n ', self.data.describe(include="all"))
            d, sort, desc
        
        with out3:
            plot = sns.pairplot(self.data, hue=None)
            plt.show()
            plot
        
        with out4:
            h = self.data.hist()
            h
#POST EDA
    def outliers(self, data, columnName, lower_bound, upper_bound):
        data = data[data[columnName]> lower_bound]
        data = data[data[columnName]< upper_bound]
        return self.data.describe()
