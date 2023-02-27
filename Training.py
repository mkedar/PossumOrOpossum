import time 
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard

# In[9]:

NAME = f'PossomOrOpossumPrediction-{int(time.time())}'

tensorboard = TensorBoard(log_dir =f'logs\\{NAME}\\')

x = pickle.load(open("C:\Users\User\Desktop\PossomOrOpossum\x.pkl",'rb'))
y = pickle.load(open("C:\Users\User\Desktop\PossomOrOpossum\y.pkl",'rb'))


# In[14]:


x = x/255 # scaling values 0-1


# In[16]:


x.shape



model = Sequential()
# layers 
model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))
# flatten layers
model.add(Flatten())
model.add(Dense(128, input_shape = x.shape[1:], activation = 'relu')) # 128 nuerons in hidden layer

model.add(Dense(2, activation = 'softmax'))


# In[25]:


model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics =['accuracy']) 


# In[26]:


model.fit(x, y, epochs=5, validation_split=0.1, batch_size=32, callbacks=[tensorboard])


# In[ ]:




