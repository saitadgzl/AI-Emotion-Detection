#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


# In[5]:


img_array = cv2.imread("Training/0/Training_36001852.jpg")


# In[6]:


img_array.shape #rgb


# In[7]:


plt.imshow(img_array) #BGR


# In[8]:


Datadirectory= "Training/" #training dataset


# In[9]:


Classes = ["0","1","2","3","4","5","6"] #list of classes


# In[10]:


for category in Classes:
    path = os.path.join(Datadirectory, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))
        plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        plt.show()
        break
    break


# In[11]:


img_size= 224 #ImageNet => 224 X 224
new_array = cv2.resize(img_array,(img_size,img_size))
plt.imshow(cv2.cvtColor(new_array, cv2.COLOR_BGR2RGB))
plt.show


# In[12]:


new_array.shape


# Read all the images and convert them to array

# In[13]:


training_Data = []   # data array

def create_training_Data():
    for category in Classes:
        path = os.path.join(Datadirectory, category)
        class_num = Classes.index(category) #0 1 label
        for img in os.listdir(path):
            try: 
                img_array = cv2.imread(os.path.join(path,img))
                new_array = cv2.resize(img_array,(img_size,img_size))
                training_Data.append([new_array,class_num])
            except Exception as e:
                pass
                


# In[14]:


create_training_Data();


# In[15]:


print(len(training_Data)) #we are just using training and deploying for real time webcam demo


# In[16]:


import random

random.shuffle(training_Data)


# In[17]:


X = [] # data / feature
y = [] # Label

for features,label in training_Data:
    X.append(features)
    y.append(label)
    
X = np.array(X).reshape(-1, img_size, img_size, 3) # converting it to 4 dimenson    


# In[18]:


X.shape


# In[19]:


#normalize the data
#X = X/225.0; #We are normalizing it.
for i in range(0,28709):

    X[i] = X[i]/255.0


# In[20]:


Y= np.array(y)


# In[21]:


Y.shape


# Deep Learning Model For Training - Transfer Learning

# In[22]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[23]:


model = tf.keras.applications.MobileNetV2() 


# In[24]:


model.summary()


# Transfer Learning - Tuning, Weights will start from last checkpoint

# In[25]:


base_input = model.layers[0].input


# In[26]:


base_output = model.layers[-2].output


# In[27]:


base_output


# In[28]:


final_output = layers.Dense(128)(base_output)  #adding new layer, after the output of global pooling layer
final_output = layers.Activation('relu')(final_output)  #activation function
final_output = layers.Dense(64)(final_output)
final_output = layers.Activation('relu')(final_output)
final_output = layers.Dense(7,activation='softmax')(final_output)  #my classes are 07, classificaion layer


# In[29]:


final_output


# In[30]:


new_model = keras.Model(inputs = base_input, outputs = final_output)


# In[31]:


new_model.summary()


# In[32]:


#new_model.compile(loss="sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])


# In[33]:


#new_model.fit(X,Y,epochs = 25) # training


# In[34]:


#new_model.save('mark4.h5')


# In[35]:


#new_model = tf.keras.models.load_model('mark4.h5')


# In[36]:


new_model.evaluate 


# In[37]:


frame = cv2.imread("happy_woman.jpg")


# In[38]:


frame.shape


# In[39]:


plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# In[40]:


# We need face detection algorithm (It works on gray images, not rgb)


# In[41]:


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# In[42]:


gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


# In[43]:


gray.shape


# In[44]:


faces = faceCascade.detectMultiScale(gray,1.1,4)
for x,y,w,h in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]
    cv2.rectangle(frame,(x,y),(x+w, y+h), (255,255,0),2)
    facess = faceCascade.detectMultiScale(roi_gray)
    if len(facess) == 0:
        print("There is no face.")
    else:
        for(ex,ey,ew,eh) in facess:
            face_roi = roi_color[ey:ey+eh, ex:ex+ew]


# In[45]:


plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# In[46]:


plt.imshow(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))


# In[47]:


final_image = cv2.resize(face_roi,(224,224))
final_image = np.expand_dims(final_image,axis=0) #need fourth dimension
final_image = final_image/225.0 #normalizing


# In[48]:


Predictions = new_model.predict(final_image)


# In[49]:


Predictions[0]


# In[50]:


np.argmax(Predictions)


# In[54]:


import cv2
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

video_path = filedialog.askopenfilename()

cap = cv2.VideoCapture(video_path)


# Kullanılacak örnek video
#video_path = "C:/Users/90539/Desktop/Face Emotion Recognition/example_video.mp4"

#cap = cv2.VideoCapture(video_path)

# Video açılmadığı takdirde verilecek hata mesajı
if not cap.isOpened():
    raise IOError("Video Açılamadı!")


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Videoyu oynat
while True:
    # Video dosyasından frame oku
    ret, frame = cap.read()

    # Videonun sonuna gelinip gelinmediğini kontrol et
    if not ret:
        break

    # Frame'i grayscale'e dönüştür
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Frame'deki yüzleri tespit et
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)


    for x, y, w, h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Yüzün etrafına dörtgen çiz
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Tahminde bulunmk için yüz verisini modele aktar
        face_roi = roi_color[ey: ey+eh, ex: ex+ew]
        final_image = cv2.resize(face_roi,(224,224))
        final_image = np.expand_dims(final_image,axis=0)
        final_image = final_image/255.0
        Predictions = new_model.predict(final_image)

        # Tahmine göre yazıyı güncelle 
        status=""
        
        if (np.argmax(Predictions)==0):
            status = "Kizgin"
        elif (np.argmax(Predictions)==1):
            status = "Igrenmis"
        elif (np.argmax(Predictions)==2):
            status = "Korkmus"
        elif (np.argmax(Predictions)==3):
            status = "Mutlu"
        elif (np.argmax(Predictions)==4):
            status = "Ifadesiz"
        elif (np.argmax(Predictions)==5):
            status = "Uzgun"
        elif (np.argmax(Predictions)==6):
            status = "Saskin"
        else:
            status = "Ifadesiz"
        
        
        x1,y1,w1,h1 = 0,0,175,75
        cv2.rectangle(frame,(x1,x1),(x1 + w1,y1 + h1),(0,0,0), -1)
        cv2.putText(frame, status, (x1 + int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255.0), 2)
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255))
        
 
    cv2.imshow("Face Emotion Recognition From Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


# In[ ]:




