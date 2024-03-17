import streamlit as st 
import streamlit_option_menu as som 
import matplotlib.pyplot as plt
from Selection.MAB import MAB_Env
import numpy as np 
import keras 
from promthmodel.turn import Autrain
import pandas as pd
import tensorflow as tf
import PIL.Image as img

class Chek : 
    
    def __init__(self , image_model:str , promth_text:str , num_arms:int):
        self.image_models = keras.models.load_model(image_model)
        self.promth_text = promth_text.lower()
        self.promth_predict = Autrain.Predicts('GTNet.pth' ,  self.promth_text)
        al_dir = f'DataGambar\{self.promth_predict[0]}'
        self.image_sl = MAB_Env(num_arms , al_dir)
        self.image_slected = self.image_sl.image_Slection()
        
    def Img_to_Vec(self , imdir): 
        imgs = tf.keras.preprocessing.image.load_img(imdir , target_size=(100,100))
        vec = tf.keras.preprocessing.image.img_to_array(imgs) / 255 
        vec = tf.expand_dims(vec , 0)
        return vec
    
    def converts(self,val:int): 
        if val == 0 : return 'Date'
        elif val == 1 : return 'kuliah'
        elif val == 2 : return 'sma'
        elif val == 3 : return 'yudo'
        
    def ImageChek(self): 
        done = True
        store = []
        hasils = []
        while done :
            counter = 0 
            for i in self.image_slected['image_dir'] : 
                print(i)
                vec = self.Img_to_Vec(i)
                hasil = self.image_models.predict(vec)
                predict = self.converts(np.argmax(hasil))
                store.append(predict.lower() == self.promth_predict)
                hasils.append(predict)
            print(hasils)
            print(store)
            counter += 1
            if False in store : 
                store.clear()
                hasils.clear()
                self.image_slected = self.image_sl.image_Slection()
            elif False not in store or counter == 5: 
                done = False 
                return hasils , self.image_sl.image_Slection()
            
cols = st.columns(3)
st.title('DEEP SEARCH FOR IMAGE')            
cols[1] = st.write('by Yudo Nidlom Firmansyah')
n_img = st.number_input('Banyak nya gambar')  
promth = st.chat_input('coba ketik : tampilkan foto saat date')
if promth :
    st.write('Sedang memilih gambar...') 
    label_list , image = Chek('GRN1.tf' , promth , n_img.as_integer_ratio()[0]).ImageChek()
    images = image['image_dir']
    r = image['ranks']
    p = image['probability']
    for i in range(len(label_list)):
        st.header(label_list[i]) 
        st.write(f'Rank : {r[i]}')
        st.write(f'Skor kecocokan : {p[i]:.2f}')
        imgsh = img.open(images[i])
        st.image(imgsh)
    

