from django.urls import path
from django.contrib.auth import views as auth_views
from django.views.i18n import set_language
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('input/', views.farm_input, name='farm_input'),
    path('recommendation/<int:recommendation_id>/', views.recommendation, name='recommendation'),
    path('about/', views.about, name='about'),
    path('login/', auth_views.LoginView.as_view(template_name='advisory/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='home'), name='logout'),
    path('signup/', views.signup, name='signup'),
    path('contact/', views.contact, name='contact'),
    path('weather/', views.weather_forecast, name='weather_forecast'),
    path('chatbot/', views.chatbot, name='chatbot'),
    path('i18n/setlang/', set_language, name='set_language'),
    path('register-shop/', views.register_shop, name='register_shop'),
    path('nearby-shops/', views.nearby_shops, name='nearby_shops_all'),
    path('nearby-shops/<str:district>/', views.nearby_shops, name='nearby_shops'),
]
