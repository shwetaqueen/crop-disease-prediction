from django.urls import path
from . import views

urlpatterns = [
	path('register/', views.registerPage, name="register"),
	path('login/', views.loginPage, name="login"),
	path('logout/', views.logoutUser, name="logout"),
	path('', views.home, name="home"),
	path("predict",views.predictImage,name='predictImage'),
	path('products/', views.products, name='products'),
	path('home2/', views.home2, name='home2'),
	path('home3/', views.home3, name='home3'),
	path('predict1/', views.predictImage1, name='predictImage1'),
	path('predict2/', views.predictImage2, name='predictImage2'),
]
