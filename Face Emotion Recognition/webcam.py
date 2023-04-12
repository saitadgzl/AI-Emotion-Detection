#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


img_array = cv2.imread("Training/0/Training_36001852.jpg")


# In[3]:


img_array.shape #rgb


# In[4]:


plt.imshow(img_array) #BGR


# In[5]:


Datadirectory= "Training/" #training dataset


# In[6]:


Classes = ["0","1","2","3","4","5","6"] #list of classes


# In[7]:


for category in Classes:
    path = os.path.join(Datadirectory, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))
        plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        plt.show()
        break
    break


# In[8]:


img_size= 224 #ImageNet => 224 X 224
new_array = cv2.resize(img_array,(img_size,img_size))
plt.imshow(cv2.cvtColor(new_array, cv2.COLOR_BGR2RGB))
plt.show


# In[9]:


new_array.shape


# Read all the images and convert them to array

# In[10]:


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
                


# In[11]:


create_training_Data();


# In[12]:


print(len(training_Data)) #we are just using training and deploying for real time webcam demo


# In[13]:


import random

random.shuffle(training_Data)


# In[14]:


X = [] # data / feature
y = [] # Label

for features,label in training_Data:
    X.append(features)
    y.append(label)
    
X = np.array(X).reshape(-1, img_size, img_size, 3) # converting it to 4 dimenson    


# In[15]:


X.shape


# In[16]:


#normalize the data
#X = X/225.0; #We are normalizing it.
for i in range(0,28709):

    X[i] = X[i]/255.0


# In[17]:


Y= np.array(y)


# In[18]:


Y.shape


# Deep Learning Model For Training - Transfer Learning

# In[19]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[20]:


model = tf.keras.applications.MobileNetV2() 


# In[21]:


model.summary()


# Transfer Learning - Tuning, Weights will start from last checkpoint

# In[22]:


base_input = model.layers[0].input


# In[23]:


base_output = model.layers[-2].output


# In[24]:


base_output


# In[25]:


final_output = layers.Dense(128)(base_output)  #adding new layer, after the output of global pooling layer
final_output = layers.Activation('relu')(final_output)  #activation function
final_output = layers.Dense(64)(final_output)
final_output = layers.Activation('relu')(final_output)
final_output = layers.Dense(7,activation='softmax')(final_output)  #my classes are 07, classificaion layer


# In[26]:


final_output


# In[27]:


new_model = keras.Model(inputs = base_input, outputs = final_output)


# In[28]:


new_model.summary()


# In[29]:


new_model.compile(loss="sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])


# In[30]:


#new_model.fit(X,Y,epochs = 25) # training


# In[31]:


#new_model.save('mark4.h5')


# In[32]:


#new_model = tf.keras.models.load_model('mark4.h5')


# In[33]:


new_model.evaluate 


# In[34]:


frame = cv2.imread("happy_woman.jpg")


# In[35]:


frame.shape


# In[36]:


plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# In[37]:


# We need face detection algorithm (It works on gray images, not rgb)


# In[38]:


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# In[39]:


gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


# In[40]:


gray.shape


# In[41]:


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


# In[42]:


plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# In[43]:


plt.imshow(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))


# In[44]:


final_image = cv2.resize(face_roi,(224,224))
final_image = np.expand_dims(final_image,axis=0) #need fourth dimension
final_image = final_image/225.0 #normalizing


# In[45]:


Predictions = new_model.predict(final_image)


# In[46]:


Predictions[0]


# In[47]:


np.argmax(Predictions)


# In[48]:


cap = cv2.VideoCapture(1)
#Webcam acik mi?
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raiseIOError("Webcam acilamadi.")
    
while True:
    ret,frame = cap.read()
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.1,4)
    for x,y,w,h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
        faces = faceCascade.detectMultiScale(roi_gray)
        if len(facess) == 0:
            print("Yuz tespit edilmedi.")
        else:
            for(ex,ey,ew,eh) in facess:
                face_roi = roi_color[ey: ey+eh, ex: ex+ew] #Yüzü kırpma
                
    final_image = cv2.resize(face_roi,(224,224))
    final_image = np.expand_dims(final_image,axis=0)  #Dördüncü boyut
    final_image = final_image/255.0
    
    Predictions = new_model.predict(final_image)
    
     
    
    if (np.argmax(Predictions)==0):
        status = "Kizgin"
        x1,y1,w1,h1 = 0,0,175,75
        #Siyah dörtgen çiz
        cv2.rectangle(frame,(x1,x1),(x1 + w1,y1 + h1),(0,0,0), -1)
        #İfadeyi yaz
        cv2.putText(frame, status, (x1 + int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255.0), 2)
        #Yüzü kare içine al
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255))
        
    elif (np.argmax(Predictions)==1):
        status = "Igrenmis"
        x1,y1,w1,h1 = 0,0,175,75
        #Siyah dörtgen çiz
        cv2.rectangle(frame,(x1,x1),(x1 + w1,y1 + h1),(0,0,0), -1)
        #İfadeyi yaz
        cv2.putText(frame, status, (x1 + int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255.0), 2)
        #Yüzü kare içine al
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255))
        
    elif (np.argmax(Predictions)==2):
        status = "Korkmus"
        xx1,y1,w1,h1 = 0,0,175,75
        #Siyah dörtgen çiz
        cv2.rectangle(frame,(x1,x1),(x1 + w1,y1 + h1),(0,0,0), -1)
        #İfadeyi yaz
        cv2.putText(frame, status, (x1 + int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255.0), 2)
        #Yüzü kare içine al
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255))
    
    elif (np.argmax(Predictions)==3):
        status = "Mutlu"
        x1,y1,w1,h1 = 0,0,175,75
        #Siyah dörtgen çiz
        cv2.rectangle(frame,(x1,x1),(x1 + w1,y1 + h1),(0,0,0), -1)
        #İfadeyi yaz
        cv2.putText(frame, status, (x1 + int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255.0), 2)
        #Yüzü kare içine al
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255))
        
    elif (np.argmax(Predictions)==4):
        status = "Ifadesiz"
        x1,y1,w1,h1 = 0,0,175,75
        #Siyah dörtgen çiz
        cv2.rectangle(frame,(x1,x1),(x1 + w1,y1 + h1),(0,0,0), -1)
        #İfadeyi yaz
        cv2.putText(frame, status, (x1 + int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255.0), 2)
        #Yüzü kare içine al
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255))
        
    elif (np.argmax(Predictions)==5):
        status = "Uzgun"
        x1,y1,w1,h1 = 0,0,175,75
        #Siyah dörtgen çiz
        cv2.rectangle(frame,(x1,x1),(x1 + w1,y1 + h1),(0,0,0), -1)
        #İfadeyi yaz
        cv2.putText(frame, status, (x1 + int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255.0), 2)
        #Yüzü kare içine al
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255))
        
    elif (np.argmax(Predictions)==6):
        status = "Saskin"
        x1,y1,w1,h1 = 0,0,175,75
        #Siyah dörtgen çiz
        cv2.rectangle(frame,(x1,x1),(x1 + w1,y1 + h1),(0,0,0), -1)
        #İfadeyi yaz
        cv2.putText(frame, status, (x1 + int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255.0), 2)
        #Yüzü kare içine al
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255))
        
    else:
        status = "İfadesiz"
        x1,y1,w1,h1 = 0,0,175,75
        #Siyah dörtgen çiz
        cv2.rectangle(frame,(x1,x1),(x1 + w1,y1 + h1),(0,0,0), -1)
        #İfadeyi yaz
        cv2.putText(frame, status, (x1 + int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255.0), 2)
        #Yüzü kare içine al
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255))
    
    cv2.imshow('Face Emotion Recognition',frame)
    
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
        

