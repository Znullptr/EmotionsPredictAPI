from django.urls import path, include
from rest_framework import routers
from . import views

router = routers.DefaultRouter()
router.register('API', views.AudiosView)
urlpatterns = [
    path('', include(router.urls))
]
