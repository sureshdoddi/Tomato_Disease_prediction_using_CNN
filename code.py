from tensorflow.compat.v1 import ConfigProto #Running TensorFlow using the CPU instead of GPU runs TensorFlow only on the CPU. By default TensorFlow uses available GPU resources to run.Use tenserflow.compat.v1.ConfigProto() to run TensorFlow using the CPU instead of GPU. A CPU (central processing unit) works together with a GPU (graphics processing unit) to increase the throughput of data and the number of concurrent calculations within an application. ... Using the power of parallelism, a GPU can complete more work in the same amount of time as compared to a CPU.

from tensorflow.compat.v1 import InteractiveSession #The only difference between Session and an InteractiveSession is that InteractiveSession makes itself the default session so that you can call run() or eval() without explicitly calling the session. 
#sess = tf.InteractiveSession()
#a = tf.constant(5.0)
#b = tf.constant(6.0)
#c = a * b
## We can just use 'c.eval()' without passing 'sess'
#print(c.eval())
#sess.close()

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2 #The first method is limiting the memory usage by percentage. So, for example, you can limit the application just only use 20% of your GPU memory. If you are using 8GB GPU memory, the application will be using 1.4 GB. (I am using Keras, so the example will be done in Keras way)
config.gpu_options.allow_growth = True #This method is using allow_growth. This method will make the application allocate only as much GPU memory based on runtime allocation. So, the application will be using the GPU memory as needed.
session = InteractiveSession(config=config)
#Transfer Learning resnet1252V2 using Keras
# import the libraries as shown below

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten #Dense layer -> Put data on different dimensions
from tensorflow.keras.models import Model
#from tensorflow.keras.applications.resnet152V2 import ResNet152V2
#from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
# re-size all the images to this

IMAGE_SIZE = [224, 224]

train_path = '/content/drive/MyDrive/tomato_data/train'
valid_path = '/content/drive/MyDrive/tomato_data/val'
#Import VGG16 library
# Here we will be using imagenet weights
import tensorflow
resnet152V2 =tensorflow.keras.applications.ResNet152V2(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
# don't train existing weights
for layer in resnet152V2.layers:
    layer.trainable = False
# useful for getting number of output classes
folders = glob('/content/drive/MyDrive/tomato_data/train/*')
# Flatten the Input
x = Flatten()(resnet152V2.output)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=resnet152V2.input, outputs=prediction)
# view the structure of the model
model.summary()
# Complie the Model
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)
# Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('/content/drive/MyDrive/tomato_data/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('/content/drive/MyDrive/tomato_data/val',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')
# fit the model
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=5,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)
import matplotlib.pyplot as plt
#Ploting Acuracy & Loss
import matplotlib.pyplot as plt
plt.plot(r.history['accuracy'])
plt.plot(r.history['val_accuracy'])
plt.plot(r.history['loss'])
plt.plot(r.history['val_loss'])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()

#Example 1

import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.load_img('/content/drive/MyDrive/tomato_data/val/Tomato___Late_blight/00ce4c63-9913-4b16-898c-29f99acf0dc3___RS_Late.B 4982.JPG', target_size = (224, 224))
imgplot = plt.imshow(test_image)
test_image = image.img_to_array(test_image)
test_image=test_image/255
test_image = np.expand_dims(test_image, axis = 0)
preds = model.predict(test_image)

preds = np.argmax(preds, axis=1)

if preds==0:
  print("Tomato - Bacterial Spot")
elif preds==1:
  print("Tomato - Early Blight")
elif preds==2:
  print("Tomato - Late Blight")
elif preds==3:
  print("Tomato - Leaf Mold")
elif preds==4:
  print("Tomato - Septoria Leaf Spot")
elif preds==5:
  print("Tomato - Spider Mites Two-spotted Spider Mite")
elif preds==6:
  print("Tomato - Target Spot")
elif preds==7:
  print("Tomato - Tomato_Yellow_Leaf_Curl_Virus")
elif preds==8:
  print("Tomato - Tomato Mosaic Virus")
else:
  print("Tomato - Healthy ")

  
#Example 2

import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.load_img('/content/drive/MyDrive/tomato_data/train/Tomato___Spider_mites Two-spotted_spider_mite/01027422-5838-4aaf-a517-01ea4e2cb6b9___Com.G_SpM_FL 9256.JPG', target_size = (224, 224))
imgplot = plt.imshow(test_image)
test_image = image.img_to_array(test_image)
test_image=test_image/255
test_image = np.expand_dims(test_image, axis = 0)
preds = model.predict(test_image)

preds = np.argmax(preds, axis=1)

if preds==0:
  print("Tomato - Bacterial Spot")
elif preds==1:
  print("Tomato - Early Blight")
elif preds==2:
  print("Tomato - Late Blight")
elif preds==3:
  print("Tomato - Leaf Mold")
elif preds==4:
  print("Tomato - Septoria Leaf Spot")
elif preds==5:
  print("Tomato - Spider Mites Two-spotted Spider Mite")
elif preds==6:
  print("Tomato - Target Spot")
elif preds==7:
  print("Tomato - Tomato_Yellow_Leaf_Curl_Virus")
elif preds==8:
  print("Tomato - Tomato Mosaic Virus")
else:
  print("Tomato - Healthy ")
