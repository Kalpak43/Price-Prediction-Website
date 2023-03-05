from django.shortcuts import render, redirect
from django.http import HttpResponse
from joblib import load
import numpy as np

model = load('G:/Dev/Projects/Price prediction/research/lr_clf.joblib')
X = load('G:/Dev/Projects/Price prediction/research/X.joblib')
V = load('G:/Dev/Projects/Price prediction/research/Variable.joblib')

# Create your views here.
def predict_price(location,sqft,bath,bhk): 
    loc_index = np.where(X.columns==location)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return model.predict([x])[0]

def home(request):
    context = {
        "locations" : V[0],
        "BHK" : V[1],
        "Bath" : V[2],
        "Min_sqft" : V[3],
        "Max_sqft" : V[4],
        "predicted" : False
    }
    
    if request.method == "POST":
        location = request.POST.get("location")
        size = int(float(request.POST.get("size")))
        bath = int(float(request.POST.get("Bath")))
        sqft = int(float(request.POST.get("sqft")))
        res = predict_price(location, sqft, bath, size)
        Pred = {
            "loc" : location,
            "size" : size,
            "bath" : bath,
            "area" : sqft,
            "res" : '%.2f' % res
        }
        context["predicted"] = True
        context["Pred"] = Pred

    return render(request, 'Base/home.html', context)
