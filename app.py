from flask import Flask, render_template, request
import requests
import pickle
import numpy as np
import sklearn
model=pickle.load(open('StoreSales.pkl','rb'))
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        Item_Weight= request.form['Item_Weight']
        Item_Fat_Content= request.form['Item_Fat_Content']
        Item_Visibility= request.form['Item_Visibility']
        Item_Type= request.form['Item_Type']
        Item_MRP= request.form['Item_MRP']
        Outlet_Size= request.form['Outlet_Size']
        Outlet_Location_Type= request.form['Outlet_Location_Type']
        Outlet_Type= request.form['Outlet_Type']
        arr=np.array([[Item_Weight,Item_Fat_Content,Item_Visibility,Item_Type,Item_MRP,Outlet_Size,Outlet_Location_Type,Outlet_Type]])
        pred= model.predict(arr)
        output=round(pred[0],2)
        if output<0:
            return render_template('index.html',prediction_texts="invalid Data")
        else:
            return render_template('index.html',prediction_text="Item Outlet Sales {}".format(output))
    else:
        return render_template('index.html')




if __name__=="__main__":
    app.run(debug=True)