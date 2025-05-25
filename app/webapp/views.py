import numpy as np
import os
from django.shortcuts import render,redirect
from django.shortcuts import get_object_or_404
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from django.forms import inlineformset_factory
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .models import *
from .forms import CreateUserForm
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.preprocessing import image
from joblib import dump, load


# Create your views here.
def registerPage(request):
    if request.user.is_authenticated:
        return redirect('home')
    else:
        form = CreateUserForm()
        if request.method == 'POST':
            form = CreateUserForm(request.POST)
            if form.is_valid():
                form.save()
                user = form.cleaned_data.get('username')
                messages.success(request, 'Account was created for ' + user)
                return redirect('login')

        context = {'form': form}
        return render(request, 'accounts/register.html', context)

def loginPage(request):
    if request.user.is_authenticated:
        return redirect('home')
    else:
        if request.method == 'POST':
            username = request.POST.get('username')
            password = request.POST.get('password')

            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('home')
            else:
                messages.info(request, 'Username OR password is incorrect')

        context = {}
        return render(request, 'accounts/login.html', context)

def logoutUser(request):
    logout(request)
    return redirect('login')

@login_required(login_url='login')
def home(request):
    return render(request, 'accounts/home3.html')

@login_required(login_url='login')
def products(request):
	return render(request, 'accounts/products.html')

@login_required(login_url='login')
def home2(request):
	return render(request, 'accounts/home2.html')

@login_required(login_url='login')
def home3(request):
	return render(request, 'accounts/home3.html')

model = load_model('trained_model_vgg16.h5')

disease_name = None
@login_required(login_url='login')
def predictImage(request):
    global disease_name
    disease_name = int(request.POST.dict()['region1'])
    return render(request, "accounts/index.html")

@login_required(login_url='login')
def predictImage2(request):
    print(disease_name,"disease_name")
    if disease_name==0:
        print("0")
        temp = int(request.POST.dict()['val1'])
        hum = int(request.POST.dict()['val2'])
        rain = int(request.POST.dict()['val3'])
        sm = int(request.POST.dict()['val4'])
        loaded_model = load('random_forest_model_grapes.pkl')
        result = loaded_model.predict([[temp,hum,rain,sm]])
        print(result)
        # if result == "Black Rot Grapes":
        #     a = "Black Rot Grapes"
        #     return render(request, "accounts/Black Rot (Grapes).html")
        # elif result == "ESCA Grapes":
        #     a = "ESCA Grapes"
        #     return render(request, "accounts/ESCA (Grapes).html")

    elif disease_name == 1:
        print("1")
        temp = int(request.POST.dict()['val1'])
        hum = int(request.POST.dict()['val2'])
        rain = int(request.POST.dict()['val3'])
        sm = int(request.POST.dict()['val4'])
        loaded_model = load('random_forest_model_maize.pkl')
        result = loaded_model.predict([[temp, hum, rain, sm]])
        print(result)

        # if result == 2:
        #     a = "Blight Maize"
        #     return render(request, "accounts/Blight (Maize).html")
        # elif result == 6:
        #     a = "Gray_Leaf Spot Maize"
        #     return render(request, "accounts/Gray Leaf Spot (Maize).html")


    elif disease_name == 2:
        print("2")
        temp = int(request.POST.dict()['val1'])
        hum = int(request.POST.dict()['val2'])
        rain = int(request.POST.dict()['val3'])
        sm = int(request.POST.dict()['val4'])
        loaded_model = load('random_forest_model_potato.pkl')
        result = loaded_model.predict([[temp, hum, rain, sm]])
        print(result)

        # if result == 9:
        #     a = "Potato Early Blight"
        #     return render(request, "accounts/Potato Early Blight.html")
        # elif result == 10:
        #     a = "Potato Late Blight"
        #     return render(request, "accounts/Potato Late Blight.html")

    elif disease_name == 3:
        print("3")
        temp = int(request.POST.dict()['val1'])
        hum = int(request.POST.dict()['val2'])
        rain = int(request.POST.dict()['val3'])
        sm = int(request.POST.dict()['val4'])
        loaded_model = load('random_forest_model_tomato.pkl')
        result = loaded_model.predict([[temp, hum, rain, sm]])
        print(result)

        # if result == 11:
        #     a = "Tomato Early Blight"
        #     return render(request, "accounts/Tomato Early Blight.html")
        # elif result == 12:
        #     a = "Tomato Late Blight"
        #     return render(request, "accounts/Tomato Late Blight.html")
        # elif result == 13:
        #     a = "Tomato Leaf Mold"
        #     return render(request, "accounts/Tomato Leaf Mold.html")


    elif disease_name == 4:
        print("4")
        temp = int(request.POST.dict()['val1'])
        hum = int(request.POST.dict()['val2'])
        rain = int(request.POST.dict()['val3'])
        sm = int(request.POST.dict()['val4'])
        loaded_model = load('random_forest_model_watermelon.pkl')
        result = loaded_model.predict([[temp, hum, rain, sm]])
        print(result)

        # if result == 0:
        #     a = 'Anthracnose Watermelon'
        #     return render(request, "accounts/Anthracnose(Watermelon).html")
        # elif result == 4:
        #     a = "Downy Mildew Watermelon"
        #     return render(request, "accounts/Downy Mildew (Watermelon).html")
    return render(request, "accounts/result.html",{"class":result[0]})

@login_required(login_url='login')
def predictImage1(request):
    fileObj = request.FILES["document"]
    fs = FileSystemStorage()
    filename = "uploaded.jpg"  # Specify the desired filename
    filePathName = fs.save(filename, fileObj)
    # filePathName = fs.save(fileObj.name, fileObj)
    filePathName = fs.url(filePathName)
    test_image = "." + filePathName
    test_image = image.load_img(test_image, target_size=(224, 224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = np.argmax(model.predict(test_image))
    result = result.item()
    print(result)

    a=''
    if result==0:
        a='Anthracnose Watermelon'
        return render(request, "accounts/Anthracnose(Watermelon).html")
    elif result==1:
        a="Black Rot Grapes"
        return render(request, "accounts/Black Rot (Grapes).html")
    elif result==2:
        a="Blight Maize"
        return render(request, "accounts/Blight (Maize).html")
    elif result==3:
        a="Common Rust Grapes"
        return render(request, "accounts/Common Rust (Grapes).html")
    elif result==4:
        a="Downy Mildew Watermelon"
        return render(request, "accounts/Downy Mildew (Watermelon).html")
    elif result==5:
        a="ESCA Grapes"
        return render(request, "accounts/ESCA (Grapes).html")
    elif result==6:
        a="Gray_Leaf Spot Maize"
        return render(request, "accounts/Gray Leaf Spot (Maize).html")
    elif result==7:
        a="Leaf Blight Grapes"
        return render(request, "accounts/Leaf Blight (Grapes).html")
    elif result==8:
        a="Mosaic Virus Watermelon"
        return render(request, "accounts/Mosaic Virus (Watermelon).html")
    elif result==9:
        a="Potato Early Blight"
        return render(request, "accounts/Potato Early Blight.html")
    elif result==10:
        a="Potato Late Blight"
        return render(request, "accounts/Potato Late Blight.html")
    elif result==11:
        a="Tomato Early Blight"
        return render(request, "accounts/Tomato Early Blight.html")
    elif result==12:
        a="Tomato Late Blight"
        return render(request, "accounts/Tomato Late Blight.html")
    elif result==13:
        a="Tomato Leaf Mold"
        return render(request, "accounts/Tomato Leaf Mold.html")

