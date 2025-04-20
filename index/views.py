from django.shortcuts import render

def index(request):
    """
    Стартовая страница с документацией API.
    """
    return render(request, 'index.html')