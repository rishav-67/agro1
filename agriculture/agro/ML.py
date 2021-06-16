import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from io import StringIO
import numpy as np
import pandas as pd
import seaborn as sns
#import tkinter as tk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
class machine_learning():
    '''
    imgdata = StringIO()
        fig.savefig(imgdata, format='svg')
        imgdata.seek(0)
        data = imgdata.getvalue()
        return data
    ''' 
    #    FOR LOADING DATASET
    df=pd.read_csv(r'crop_production.csv')
    def headc(self):
        df1=self.df  
        arr=self.count_state()
        df4=df1.head(1)
        
        for i in arr:
             df3 = df1[df1['State_Name'] == i]
             zz=df3.head(10)
             
             df4=df4.append(zz, ignore_index = True)
             
        return df4

    #  FOE SEEING DATASET
    def headx(self):   
        df1=self.df               
        
        df1.replace([np.inf, -np.inf], np.nan, inplace=True)
        df6=df1.dropna()
        des=df6.describe()
        #des=df1['State_Name'].unique()
        print(des)
        return des
    def count_state(self):
        df1=self.df['State_Name'].unique()
        
        return df1
    #   NO OF STATES/UT AND SEASON 

    def count_state_ut(self):
        df1=self.df['State_Name'].unique()
        df2=self.df['Season'].unique()
        df3=[df1,df2]
        return df3

    # SEE INFO OF YOUR DATASET

    def info(self):
        df1=[]
        df1.append(self.df.info())
        return df1

    # FREQUENCY OF STATE_NAME IN DATASET TO PREDICT THE COUNT OF EACH STATE

    def fre_state_ut(self):
        
        df1=self.df['State_Name'].unique()
        return df1
    def fre_state_sea(self):
        
        df1=self.df['Season'].unique()
        return df1
    
    def dummy(self):
        df1=self.df['District_Name'].unique()
        return df1  
    def fre_crop(self):
        df8=self.df
        df6=df8.dropna()
        df = df6.groupby(by='Crop')['Production'].sum().reset_index().sort_values(by='Production', ascending=False).head(7)
        df11=df['Crop'].unique()
        return df11
    #ADD DISTRICT IN FORM HERE FILTER STATE WISE-DISTRICT
    def state_district(self):
        #df3=self.df['State_Name'].unique()
        d=self.df['State_Name'].unique()
        pair={}
        for i in d:
            df41=self.df[self.df['State_Name'] == i]
            df8=str(df41['District_Name'].unique())
            a=df8.replace(" ","|")
            #y=list(y)
            p=""
            for i in range(0,len(a)):
	            if a[i]=="'":
		            continue
	            else:
		            p+=a[i]


            
            pair[i]=p
            
        return(pair)
    def crop_macro(self,a):
        
        oilseeds=['other oilseeds','Safflower','Niger seed','Castor seed','Linseed','Sunflower','Rapeseed &Mustard','Sesamum','Oilseeds total']
        Nuts=['Arcanut (Processed)','Atcanut (Raw)','Cashewnut Processed','Cashewnut Raw','Cashewnut','Arecanut','Groundnut']
        Commercial=['Tobacco','Coffee','Tea','Sugarcane','Rubber']
        Beans=['Bean','Lab-Lab','Moth','Guar seed','Soyabean','Horse-gram']
        fibres=['other fibres','Kapas','Jute & mesta','Jute','Mesta','Cotton(lint)','Sannhamp']
        Cereal=['Rice','Maize','Wheat','Barley','Varagu','Other Cereals & Millets','Ragi','Small millets','Bajra','Jowar', 'Paddy','Total foodgrain','Jobster']
        Pulses=['Moong','Urad','Arhar/Tur','Peas & beans','Masoor','Other Kharif pulses','other misc. pulses','Ricebean (nagadal)','Rajmash Kholar','Lentil','Samai','Blackgram','Korra','Cowpea(Lobia)','Other  Rabi pulses','Other Kharif pulses','Peas & beans (Pulses)','Pulses total','Gram']
        spices=['Perilla','Ginger','Cardamom','Black pepper','Dry ginger','Garlic','Coriander','Turmeric','Dry chillies','Cond-spcs other']
        Fruits=['Peach','Apple','Litchi','Pear','Plums','Ber','Sapota','Lemon','Pome Granet','Other Citrus Fruit','Water Melon','Jack Fruit','Grapes','Pineapple','Orange','Pome Fruit','Citrus Fruit','Other Fresh Fruits','Mango','Papaya','Coconut','Banana']
        
        Vegetables=['Turnip','Peas','Beet Root','Carrot','Yam','Ribed Guard','Ash Gourd ','Pump Kin','Redish','Snak Guard','Bottle Gourd','Bitter Gourd','Cucumber','Drum Stick','Cauliflower','Beans & Mutter(Vegetable)','Cabbage','Bhindi','Tomato','Brinjal','Khesari','Sweet potato','Potato','Onion','Tapioca','Colocosia']
           
        if a in oilseeds: 
            return 'oilseeds'     
        if a in Nuts:
            return 'Nuts'
        if a in Commercial:
            return 'Commercial'
        if a in Beans:
            return 'Beans'
        if a in fibres:
            return 'fibres'
        if a in Cereal:
            return 'Cereal'
        if a in Pulses:   
            return 'Pulses'
        if a in spices:
            return 'spices'
        if a in Fruits:
            return 'Fruits'
        if a in Vegetables:
            return 'Vegetables'
    def apply_crop(self):
        df5=self.df
        self.df['crop_category']=self.df['Crop'].apply(self.crop_macro)
        #df1=self.df.head()
        #df2=df1['crop_category']
        return self.df.head()
    def best(self,df88,y,df99):
        a=[]
        xtrain,xtest,ytrain,ytest=train_test_split(df88,y,test_size=0.2)

        model1=LinearRegression()
        model1.fit(xtrain,ytrain)
        pred=model1.predict(df99)
        a.append([model1.score(xtest,ytest)*100,'LinearRegression',pred])
        print("li")

        model2=linear_model.Lasso(alpha=50, max_iter=20, tol=0.1)
        model2.fit(xtrain,ytrain)     
        pred=model2.predict(df99)
        a.append([model2.score(xtest,ytest)*100,'Lasso',pred])
        print("las")

        model3=RandomForestRegressor(random_state=5)
        model3.fit(xtrain,ytrain)
        pred=model3.predict(df99)
        a.append([model3.score(xtest,ytest)*100,'RandomForestRegressor',pred])
        print("reg")
        
        params = {'n_neighbors':[2,3,4]}
        knn = neighbors.KNeighborsRegressor()
        model5=GridSearchCV(knn, params, cv=5)
        model5.fit(xtrain,ytrain)
        pred=model5.predict(df99) 
        a.append([model5.score(xtest,ytest)*100,'KNeighborsRegressor',pred])
        print("knn")
        model6=Ridge(alpha=50, max_iter=20, tol=0.1)
        model6.fit(xtrain,ytrain)
        pred=model6.predict(df99)
        a.append([model6.score(xtest,ytest)*100,'Ridge',pred])     
        print("ri")
        return a
    def predict(self,st,cir,cr,cryear,ar,sea):
               
        df77=self.df
        #data_rice = df77[df77['State_Name'] == st]
        data_rice = df77[df77['Crop'] == cr]
        data_rice.replace([np.inf, -np.inf], np.nan, inplace=True)
        df6=data_rice.dropna()
        #df6['crop_category']=df6['Crop'].apply(self.crop_macro)
        #Andaman and Nicobar Islands	NICOBARS	2000	Kharif	Rice	102.00	321.00
        #West Bengal	PURULIA	2014	Winter	Rice	279151.00	597899.00	
        df7=df6[['State_Name','District_Name','Crop_Year','Season','Crop','Area']]
        df8=df6.Production
        '''st='West Bengal'
        ci='PURULIA'
        cr='Rice'
        cryear=2014
        ar=279151.00
        sea='Winter'
        return 99'''
        print(st,cir,cr,cryear,ar,sea)
        df11 = pd.DataFrame({"State_Name":[st],
                    "District_Name":[cir],
                    "Crop_Year":[cryear],
                    "Season":[sea],
                    "Crop":[cr],
                    "Area":[ar]
                    
                    
                    })   
        
        print(df11)
        df55=df7.append(df11, ignore_index = True)
        df777=df55.copy()
        ab=LabelEncoder()
        df777['Season_n']=ab.fit_transform(df777['Season'])
        df777['State_Name_n']=ab.fit_transform(df777['State_Name'])
        df777['District_Name_n']=ab.fit_transform(df777['District_Name'])
        
        df88=df777.drop(['Season','State_Name','District_Name','Crop'],axis="columns")
        df99 = df88.iloc[[-1]]
        print(df99)
        q=df88.drop(df88.tail(1).index,inplace=True)
        y = df6['Production']
        abc=[]
        pp=self.best(df88,y,df99)
        max1=0.00
        index=0
        for i in range(0,len(pp)):
            z=pp[i][0]
            if z>max1:
                max1=z
                index=i
        dq=[pp,pp[index]]
        return dq
       
        

    def year(self):
        data_rice_dropped = self.df[self.df['Crop_Year']<=2020]
        p=data_rice_dropped.Crop_Year.unique()

        return p
    def lineplot(self,df777):
        #'State_Name','District_Name','Crop_Year','Season',
        
        
        ab=LabelEncoder()
 
        df777['Season_n']=ab.fit_transform(df777['Season'])
        season_mapping = dict(zip(ab.classes_, ab.transform(ab.classes_)))
        
        df777['District_Name_n']=ab.fit_transform(df777['District_Name']) 
        district_mapping = dict(zip(ab.classes_, ab.transform(ab.classes_)))
        df88=df777.drop(['Season','State_Name','District_Name','Crop'],axis="columns")
        dis_map=[]
        for key,value in district_mapping.items():
            dis_map.append(key)
        sea_map=[]
        for key,value in season_mapping.items():
            sea_map.append(key)
        
        inpu=['District_Name_n','Crop_Year','Season_n','Area']
        x=[]
        #xx=['a','b','c']
        for i in range(0,len(inpu)):
            fig = plt.figure(figsize=(6,6))
            sns.lineplot(x=inpu[i],y='Production',data=df88,ci=None)
            if inpu[i]=="District_Name_n":
                plt.xticks([i for i in range(len(dis_map))],dis_map,rotation=90)
                plt.tight_layout()
            if inpu[i]=="Season_n":
                plt.xticks([i for i in range(len(sea_map))],sea_map,rotation=90)
                plt.tight_layout()
            imgdata = StringIO()
            fig.savefig(imgdata, format='svg')
            imgdata.seek(0)
            data = imgdata.getvalue()
            x.append(data)
        print(type(district_mapping))
        return x
    def main_lineplot(self,sta,cro):
        df8=self.df
        df6=df8[df8['State_Name'] == sta]
        df777=df6.dropna()
        
        
        df6=df777[df777['Crop'] == cro]
        second=self.lineplot(df6)
        
        return second
    def main_lineplot1(self,sta):
        df8=self.df
        df6=df8[df8['State_Name'] == sta]
        df777=df6.dropna()
        
        first=self.lineplot(df777)
        return first
    def histo(self,df777):
        
        ab=LabelEncoder()
 
        df777['Season_n']=ab.fit_transform(df777['Season'])
        season_mapping = dict(zip(ab.classes_, ab.transform(ab.classes_)))
        
        df777['District_Name_n']=ab.fit_transform(df777['District_Name']) 
        district_mapping = dict(zip(ab.classes_, ab.transform(ab.classes_)))
        df88=df777.drop(['Season','State_Name','District_Name','Crop'],axis="columns")
        dis_map=[]
        for key,value in district_mapping.items():
            dis_map.append(key)
        sea_map=[]
        for key,value in season_mapping.items():
            sea_map.append(key)
        
        inpu=['District_Name_n','Crop_Year','Season_n','Area']
        xx=[]
        
        for i in range(0,len(inpu)):
            se=inpu[i]
            print(se)
            
            
            fig = plt.figure(figsize=(6,6))
            
            se=inpu[i]
            sns.set()
            sns.distplot(df88[se],hist_kws={'edgecolor':'yellow','linewidth':5,'linestyle':'--','alpha':0.5},bins=10)
            if inpu[i]=="District_Name_n":
                plt.xticks([i for i in range(len(dis_map))],dis_map,rotation=90)
                plt.tight_layout()
            if inpu[i]=="Season_n":
                plt.xticks([i for i in range(len(sea_map))],sea_map,rotation=90)
                plt.tight_layout()
            imgdata = StringIO()
            fig.savefig(imgdata, format='svg')
            imgdata.seek(0)

            data = imgdata.getvalue()
            xx.append(data)
        return xx
    def main_histplot(self,sta,cro):
        df8=self.df
        df6=df8[df8['State_Name'] == sta]
        df777=df6.dropna()
        
        
        df6=df777[df777['Crop'] == cro]
        second=self.histo(df6)
        
        return second
    def main_histplot1(self,sta):
        df8=self.df
        df6=df8[df8['State_Name'] == sta]
        df777=df6.dropna()
        
        first=self.histo(df777)
        return first

    def barplot(self,df6):
        
        a=['District_Name','Crop_Year','Season','Area']
        xx=[]
        for i in range(0,len(a)):
            se=a[i]
            print(se)
            df = df6.groupby(by=se)['Production'].sum().reset_index().sort_values(by='Production', ascending=False).head(10)
            fig = plt.figure(figsize=(6,6))
            se=a[i]
            sns.barplot(df[se], df.Production,errwidth=0)
            if a[i]=="District_Name":
                plt.xticks(rotation=90)
                plt.tight_layout()
            imgdata = StringIO()
            fig.savefig(imgdata, format='svg')
            imgdata.seek(0)

            data = imgdata.getvalue()
            xx.append(data)
        return xx
    def main_barplot(self,sta,cro):
        df8=self.df
        df6=df8[df8['State_Name'] == sta]
        df777=df6.dropna()
        
        
        df6=df777[df777['Crop'] == cro]
        second=self.barplot(df6)
        
        return second
    def main_barplot1(self,sta):
        df8=self.df
        df6=df8[df8['State_Name'] == sta]
        df777=df6.dropna()
        
        first=self.barplot(df777)
        return first
    def scatter(self,df777):
        ab=LabelEncoder()
 
        df777['Season_n']=ab.fit_transform(df777['Season'])
        season_mapping = dict(zip(ab.classes_, ab.transform(ab.classes_)))
        
        df777['District_Name_n']=ab.fit_transform(df777['District_Name']) 
        district_mapping = dict(zip(ab.classes_, ab.transform(ab.classes_)))
        df88=df777.drop(['State_Name','District_Name','Crop'],axis="columns")
        dis_map=[]
        for key,value in district_mapping.items():
            dis_map.append(key)
        sea_map=[]
        for key,value in season_mapping.items():
            sea_map.append(key)
        
        a=['District_Name_n','Crop_Year','Season_n','Area']
        xx=[]
       
        for i in range(0,len(a)):
            se=a[i]
            print(se)
            
            #df = df6.groupby(by=se)['Production'].sum().reset_index().sort_values(by='Production', ascending=False).head(10)
            fig = plt.figure(figsize=(6,6))
            se=a[i]
            sns.scatterplot(x=se,y='Production',data=df88,hue='Season',ci=None)
            if a[i]=="District_Name_n":
                plt.xticks([i for i in range(len(dis_map))],dis_map,rotation=90)
                plt.tight_layout()
            if a[i]=="Season_n":
                plt.xticks([i for i in range(len(sea_map))],sea_map,rotation=90)
                plt.tight_layout()
            imgdata = StringIO()
            fig.savefig(imgdata, format='svg')
            imgdata.seek(0)

            data = imgdata.getvalue()
            xx.append(data)
        return xx
    def main_scatter(self,sta,cro):
        df8=self.df
        df6=df8[df8['State_Name'] == sta]
        df777=df6.dropna()
        
        
        df6=df777[df777['Crop'] == cro]
        second=self.scatter(df6)
        
        return second
    def main_scatter1(self,sta):
        df8=self.df
        df6=df8[df8['State_Name'] == sta]
        df777=df6.dropna()
        
        first=self.scatter(df777)
        return first
    def heat(self):
        df8=self.df
        df6=df8.dropna()
        fig = plt.figure(figsize=(12,4))
        
        sns.heatmap(df6.corr(),annot=True)
        imgdata = StringIO()
        fig.savefig(imgdata, format='svg')
        imgdata.seek(0)
    
        data = imgdata.getvalue()
        return data
     
    def pie(self,df6):
        a=['District_Name','Crop_Year','Season']
        xx=[]
        for i in range(0,len(a)):
            se=a[i]
            
            segment = df6[se].value_counts()
            segment_label = df6[se].unique()
            fig = plt.figure(figsize=(8,6))
            se=a[i]
            plt.pie(segment,
            autopct = '%1.1f%%',
            labels = segment_label,
            
            shadow = True,
            )
            plt.tight_layout()
            imgdata = StringIO()
            fig.savefig(imgdata, format='svg')
            imgdata.seek(0)
       
            data = imgdata.getvalue()
            xx.append(data)
        return xx
    def main_pie(self,sta,cro):
        df8=self.df
        df6=df8[df8['State_Name'] == sta]
        df777=df6.dropna()
        
        
        df6=df777[df777['Crop'] == cro]
        second=self.pie(df6)
        
        return second
    def main_pie1(self,sta):
        df8=self.df
        df6=df8[df8['State_Name'] == sta]
        df777=df6.dropna()
        
        first=self.pie(df777)
        return first