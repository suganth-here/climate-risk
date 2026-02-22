from django.urls import path

from . import views


urlpatterns = [
    path("metadata/", views.metadata_view, name="metadata"),
    path("predict/", views.predict_view, name="predict"),
    path("portfolio/analyze/", views.portfolio_analyze_view, name="portfolio_analyze"),
]
