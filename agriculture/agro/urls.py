from django.urls import path,include
from . import views       
from django.conf import settings
from django.conf.urls.static import static 
                                   
#from django.u  
urlpatterns = [
    path('',views.index,name="iskahome"),
    path('predict/',views.show,name="iskashow"),
    path('graphs/',views.graph,name="iskagraph"), 
    path('graphscommon/',views.graphcommon,name="iskagraphco"),
    path('datas/',views.headgo,name="iskahead")
           
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)