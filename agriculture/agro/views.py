from django.shortcuts import render                 
import django 
from .forms import select_check
from .ML import machine_learning 
import json
# Create your views here.     
def index(request):  
    
    
    return render(request,'agro/index.html')
def show(request):
    
    if request.method=="POST":
        #print(request)
        ap=request.POST['area']  
        b=request.POST['state']
        cx=request.POST['city']
        x=request.POST['select_crops']
        y=request.POST['year']
        z=request.POST['crops']
        r=request.POST['season_ne']     
        print(b,cx,z,y,ap,r)  
        c=machine_learning()
        p,r=c.predict(b,cx,z,y,ap,r)                
        q=c.fre_state_sea()
        #print(p)
        ax1=r[0]
        ax2=r[1]
        ax3=r[2]
        a={'a':p,'cc':q,'ax1':ax1,'ax2':ax2,'ax3':ax3}
        #p=[]
        return render(request,'agro/predictionresult.html',a) 
    else:
        c=machine_learning()
        
        q=c.fre_state_sea()
        #print(p)   
        a={'cc':q}

        return render(request,'agro/form.html',a) 
    

def graph(request):  
    if request.method=="POST":
        gra=request.POST['drop1']
        sta=request.POST['drop16']  
        cro=request.POST['drop162']
        if gra=="LINEPLOT": 
            pp=machine_learning()
            xp=pp.main_lineplot(sta,cro)
            #xc=pp.lineplotone(sta,cro) 
            ax=['DISTRICT-PRODUCTION LINEPLOT','CROPYEAR-PRODUCTION LINEPLOT','SEASON-PRODUCTION LINEPLOT','AREA-PRODUCTION LINEPLOT']
            bx=zip(xp,ax)


            #a={'a':xp}
            
            a={'aa':bx}
            return render(request,'agro/graphone.html',a)

        if gra=="BARGRAPH": 
            pp=machine_learning()
            xp=pp.main_barplot(sta,cro)
            ax=['DISTRICT-PRODUCTION BARGRAPH','CROPYEAR-PRODUCTION BARGRAPH','SEASON-PRODUCTION BARGRAPH','AREA-PRODUCTION BARGRAPH']
            bx=zip(xp,ax)


            #a={'a':xp}
            
            a={'aa':bx}
            return render(request,'agro/graphone.html',a)
        if gra=="HISTOGRAM":
            pp=machine_learning()
            xp=pp.main_histplot(sta,cro)  
            ax=['DISTRICT-PRODUCTION HISTOGRAM','CROPYEAR-PRODUCTION HISTOGRAM','SEASON-PRODUCTION HISTOGRAM','AREA-PRODUCTION HISTOGRAM']
            bx=zip(xp,ax)
            a={'aa':bx}
            return render(request,'agro/graphone.html',a)
        if gra=="SCATTER PLOTTING":
            pp=machine_learning()
            xp=pp.main_scatter(sta,cro)
            ax=['DISTRICT-PRODUCTION SCATTER PLOTTING','CROPYEAR-PRODUCTION SCATTER PLOTTING','SEASON-PRODUCTION SCATTER PLOTTING','AREA-PRODUCTION SCATTER PLOTTING']
            bx=zip(xp,ax)
            a={'aa':bx}     
            return render(request,'agro/graphone.html',a)
        if gra=="PIE-CHART":   
            pp=machine_learning()
            xp=pp.main_pie(sta,cro)
            ax=['DISTRICT-PRODUCTION PIE-CHART','CROPYEAR-PRODUCTION PIE-CHART','SEASON-PRODUCTION PIE-CHART']
            bx=zip(xp,ax) 
            a={'aa':bx}
            return render(request,'agro/graphone.html',a)
        if gra=="HEATMAP": 
            pp=machine_learning()
            xp=pp.heat()
            
            a={'aaq':xp}
            
            return render(request,'agro/test.html',a)

    else:
        pp=machine_learning()
        xp=pp.fre_state_ut() 
        xq=pp.fre_crop()  
        a={'a':xp,'b':xq}
        return render(request,'agro/graph.html',a) 
def graphcommon(request):
    if request.method=="POST":
        print(request)
        gra=request.POST['drop181'] 
        sta=request.POST['drop182'] 
        #sta5=request.POST['drop16']
        if gra=="LINEPLOT": 
            pp=machine_learning()
            xp=pp.main_lineplot1(sta)
            ax=['DISTRICT-PRODUCTION LINEPLOT','CROPYEAR-PRODUCTION LINEPLOT','SEASON-PRODUCTION LINEPLOT','AREA-PRODUCTION LINEPLOT']
            bx=zip(xp,ax)


            #a={'a':xp}
            
            a={'aa':bx}
            return render(request,'agro/graphone.html',a)

        if gra=="BARGRAPH": 
            pp=machine_learning()
            xp=pp.main_barplot1(sta)
            ax=['DISTRICT-PRODUCTION BARGRAPH','CROPYEAR-PRODUCTION BARGRAPH','SEASON-PRODUCTION BARGRAPH','AREA-PRODUCTION BARGRAPH']
            bx=zip(xp,ax)


            #a={'a':xp}
            
            a={'aa':bx}
            return render(request,'agro/graphone.html',a)
        if gra=="HISTOGRAM":
            pp=machine_learning()
            xp=pp.main_histplot1(sta)
            ax=['DISTRICT-PRODUCTION HISTOGRAM','CROPYEAR-PRODUCTION HISTOGRAM','SEASON-PRODUCTION HISTOGRAM','AREA-PRODUCTION HISTOGRAM']
            bx=zip(xp,ax)
            a={'aa':bx}  
            return render(request,'agro/graphone.html',a)
        if gra=="SCATTER PLOTTING":
            pp=machine_learning()
            xp=pp.main_scatter1(sta)
            ax=['DISTRICT-PRODUCTION SCATTER PLOTTING','CROPYEAR-PRODUCTION SCATTER PLOTTING','SEASON-PRODUCTION SCATTER PLOTTING','AREA-PRODUCTION SCATTER PLOTTING']
            bx=zip(xp,ax)
            a={'aa':bx}
            return render(request,'agro/graphone.html',a)
        if gra=="PIE-CHART": 
            pp=machine_learning()
            xp=pp.main_pie1(sta)
            ax=['DISTRICT-PRODUCTION LINEPLOT','CROPYEAR-PRODUCTION LINEPLOT','SEASON-PRODUCTION LINEPLOT']
            bx=zip(xp,ax) 
            a={'aa':bx}
            return render(request,'agro/graphone.html',a)
        if gra=="HEATMAP": 
            pp=machine_learning()
            xp=pp.heat()
            a={'aaq':xp}  
            
            return render(request,'agro/test.html',a)


    else: 
        pp=machine_learning()
        xp=pp.fre_state_ut()
        xq=pp.fre_crop()  
        a={'a':xp,'b':xq}    
        return render(request,'agro/graphcommon.html',a)  

def headgo(request):
    pp=machine_learning()
    xp=pp.headc()

    json_records = xp.reset_index().to_json(orient ='records')
    data = []
    data = json.loads(json_records)

    a={'a':data}
    return render(request,'agro/head.html',a)  