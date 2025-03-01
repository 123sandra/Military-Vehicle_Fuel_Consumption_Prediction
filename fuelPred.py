import streamlit as st
import sklearn
import pickle
from PIL import Image
def main():
    st.title("Military Vehicle Fuel Consumption Prediction")
    st.markdown("<h4 style='color:Green;'>Designed for military logistics, this application forecasts fuel usage for vehicles in varied terrains. It leverages machine learning to analyze weight, horsepower, origin etc. Accurate predictions support efficient mission planning and resource allocation.</h4>", unsafe_allow_html=True)
    img=Image.open("Dramatic_War_Landscape.jpg")
    st.image(img,width=600)
    cylinders=st.text_input("Number of Cylinders (Cylinders can range from 2 to 8)")
    displacement=st.text_input("Displacement")
    horsepower=st.text_input("HorsePower of the vehicle")
    weight=st.text_input("Weight of the vehicle in Kg")
    acceleration=st.text_input("Acceleration")
    mode_year=st.slider("Model's Year (Year ranges from 1970-90 insert 70-90)",70,90)
    origin=st.radio("Origin of the vehicle 1-US 2-Europe 3-Japan",["1","2","3"],horizontal=True)
    terrain=st.text_input("Terrain (0-Road,1-Desert,2-Forest,3-Snow)")
    features=[cylinders,displacement,horsepower,weight,acceleration,mode_year,origin,terrain]
    scaler=pickle.load(open('scaler.sav','rb'))
    model=pickle.load(open('model.sav','rb'))
    pred=st.button("Predict")
    if pred:
        result=model.predict(scaler.transform([features]))
        st.write("<h3>Predicted Fuel Requirement per Kilometer:</h3>",unsafe_allow_html=True)
        st.write(result,"Liters per Km")
    img2=Image.open('Flux_Dev_A_dramatic_scene_depicting_a_diverse_range_of_militar_3.jpeg')
    st.image(img2,width=650)
main()
