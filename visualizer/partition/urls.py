from django.urls import path
from . import views

app_name = 'partition'
urlpatterns = [
    path('', views.index, name='index'),
    path('submitTxt', views.submitTxt, name='submitTxt')
]
