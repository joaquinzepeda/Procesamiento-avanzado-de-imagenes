# -*- coding: utf-8 -*-
"""Tarea5_imagenes.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Gko2mgN7qFImaH2lCugIoRudM07G1Xph

# Desarrollo por Joaquin Zepeda V.

Tarea 5 EL7008 - Clasificación de objetos usando CNNs.

El objetivo de esta tarea es implementar un sistema de clasificación de objetos usando redes neuronales
convolucionales (CNNs). En esta tarea se debe usar la librería pytorch para poder generar los tensores que
corresponden a las imágenes y sus labels (etiquetas), además de implementar arquitecturas de red y códigos
para entrenamiento y evaluación
"""

!wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

!tar -xzvf cifar-10-python.tar.gz

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy
from torchvision import datasets, models, transforms
import time
import os
import random

SEED = 1234


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = ('cuda' if torch.cuda.is_available() else 'cpu')

device

"""# Sección 1. Implementar el código para que pytorch acceda a los datasets, para el conjunto de entrenamiento, validación y prueba."""

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

L = unpickle("/content/cifar-10-batches-py/data_batch_1")

data,labels,filenames = L['data'],L['labels'],L['filenames']
metadata = unpickle("/content/cifar-10-batches-py/batches.meta")
label_names = metadata['label_names']

#he first 1024 entries contain the red channel values, the next 1024 the green, 
#and the final 1024 the blue. The image is stored in row-major order, so that the 
#first 32 entries of the array are the red channel values of the first row of the image.
data[0]

from google.colab.patches import cv2_imshow


fig=plt.figure(figsize=(10, 10))
for i in range(1, 26):
    fig.add_subplot(5, 5, i)
    img = np.reshape(data[i], (3,32,32)).transpose(1,2,0)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.title(label_names[labels[i]])
plt.show()

def escalamiento(x):
    return -1 + 2/255*x

x = np.arange(0,256,1)
plt.title("Escalamiento lineal")
plt.plot(x,escalamiento(x))
plt.xlabel("Valor del pixel original")
plt.ylabel("Valor del pixel escalado")
plt.grid()



from torch.utils.data import Dataset

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


class CIFAR10Train(Dataset):
    def __init__(self, path):
        # Constructor, debe leer el archivo data_batch_1 dentro de la carpeta
        # indicada (este archivo se usará para el set de entrenamiento)
        self.dict_data = unpickle(path+'data_batch_1')
        self.labels,self.filenames = self.dict_data['labels'],self.dict_data['filenames']

    def __len__(self):
        # Debe retornar el número de imágenes en el dataset de entrenamiento
        return len(self.filenames)

    def __getitem__(self, index):
        data = self.dict_data['data']
        # Debe retornar un par label, image
        # Donde label es una etiqueta, e image es un arreglo de 3x32x32
        # index es un número (o lista de números) que indica cuáles imágenes
        # y labels se deben retornar
        #escalamineto lineal
        datax = -1 +2/255*data[index]
        img = np.reshape(datax, (3,32,32))
        return self.labels[index], img

class CIFAR10Val(Dataset):
    def __init__(self, path):
        # Constructor, debe leer el archivo data_batch_2 dentro de la carpeta
        # indicada (este archivo se usará para el set de entrenamiento)
        self.dict_data = unpickle(path+'data_batch_2')
        self.labels,self.filenames = self.dict_data['labels'],self.dict_data['filenames']

    def __len__(self):
        # Debe retornar el número de imágenes en el dataset de entrenamiento
        return len(self.filenames)

    def __getitem__(self, index):
        data = self.dict_data['data']
        # Debe retornar un par label, image
        # Donde label es una etiqueta, e image es un arreglo de 3x32x32
        # index es un número (o lista de números) que indica cuáles imágenes
        # y labels se deben retornar
        #escalamineto lineal
        datax = -1 +2/255*data[index]
        img = np.reshape(datax, (3,32,32))
        return self.labels[index], img

class CIFAR10Test(Dataset):
    def __init__(self, path):
        # Constructor, debe leer el archivo test_batch
        # indicada (este archivo se usará para el set de entrenamiento)
        self.dict_data = unpickle(path+'test_batch')
        self.labels,self.filenames = self.dict_data['labels'],self.dict_data['filenames']

    def __len__(self):
        # Debe retornar el número de imágenes en el dataset de entrenamiento
        return len(self.filenames)

    def __getitem__(self, index):
        data = self.dict_data['data']
        # Debe retornar un par label, image
        # Donde label es una etiqueta, e image es un arreglo de 3x32x32
        # index es un número (o lista de números) que indica cuáles imágenes
        # y labels se deben retornar

        #escalamineto lineal
        datax = -1 +2/255*data[index]
        img = np.reshape(datax, (3,32,32))
        return self.labels[index], img

trainDataset = CIFAR10Train("/content/cifar-10-batches-py/")
valDataset = CIFAR10Val("/content/cifar-10-batches-py/")
testDataset = CIFAR10Test("/content/cifar-10-batches-py/")

BATCH_SIZE = 256
train_loader = torch.utils.data.DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = torch.utils.data.DataLoader(valDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader= torch.utils.data.DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

"""# Sección 2. Implementar una red inicial.

 Las primeras capas convolucionales que se recomienda usar están indicadas en el enunciado. Elija un tamaño N de neuronas en la última capa oculta que le parezca
apropiado. Se recomienda usar max pooling cada cierta cantidad de capas para reducir el tamaño
espacial de los tensores. Elija un batch_size inicial que le parezca apropiado.
"""

class MyNet(nn.Module):
 def __init__(self, N=128):
    super(MyNet, self).__init__()
    self.nclasses = 10
    #nn.Conv2d(in_channels, out_channels, kernel_size)
    self.conv1 = nn.Conv2d(3, 64, 3, padding = 1) #64 filtros de 3x3, 3 canales de entrada
    self.conv2 = nn.Conv2d(64, 64, 3, padding = 1) 
    self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)

    self.bn1 = torch.nn.BatchNorm2d(64)
    self.bn2 = torch.nn.BatchNorm2d(64)
    self.bn3 = torch.nn.BatchNorm2d(128)

    self.MaxPool = nn.MaxPool2d(2, 2)

    self.fc1 = nn.Linear(8192, 1024)
    self.fc2 = nn.Linear(1024, 24*N)
    self.fc3 = nn.Linear(24*N, N)
    self.fc_last = nn.Linear(N, self.nclasses)
 def forward(self, x):
    x = self.bn1(F.relu(self.conv1(x)))
    x = self.MaxPool(self.bn2(F.relu(self.conv2(x))))
    x = self.MaxPool(self.bn3(F.relu(self.conv3(x))))

    #transformamos el tensor de una capa convolucional a una capa fully connected
    x = x.view(x.size()[0], -1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = self.fc_last(x)
    return x

"""## Sección 2.a Entrenar la red usando el conjunto de entrenamiento y controlando el sobreajuste con el conjunto de validación.


La red puede sufrir sobreajuste si la cantidad de parámetros es grande, a medida que el entrenamiento va
progresando. Para poder evitar el sobreajuste, se recomienda usar un enfoque basado en patience. Además,
se debe ir guardando el menor loss de validación, se debe ir guardando checkpoints cada vez que el loss
actual es menor que el menor loss de validación existente. Posteriormente, para poder evaluar el
desempeño de la red, se debe recuperar el mejor checkpoint almacenado
"""

# Parte de los prints de los accuracy's y de los loss se basaron en modelos
# del curso Deep learning que estoy cursando actualmente.
def train(net, optimizer, num_epocas):
  inicio = time.time()
  #copiamos el modelo utilizando la libreria copy
  best_model_wts = copy.deepcopy(net.state_dict()) 
  train_losses = []
  train_counter = []
  train_accuracy = []
  val_losses = []
  val_accuracy = []
  best_acc = 0.0
  best_loss = 2e32
  for epoch in range(num_epocas):
    print('Epoch {}/{}'.format(epoch, num_epocas-1))
    print('-' * 10)

    net.train() #Modo entrenamiento

    running_loss = 0.0
    running_corrects = 0.0
    for i, data in enumerate(train_loader, 0): # Obtener batch
        labels = data[0].cuda()
        inputs = data[1].cuda().float()
        optimizer.zero_grad()
        outputs = net(inputs) #salidas de la red
        preds = outputs.argmax(axis=1) #predicciones
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss /len(train_loader.dataset) #promedio de error 
    epoch_acc = running_corrects.double() / len(train_loader.dataset) #promedio de accuracy
    train_losses.append(epoch_loss)
    train_counter.append(epoch)
    train_accuracy.append(epoch_acc)
      
    print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    #Validacion 
    net.eval()

    running_loss = 0.0
    running_corrects = 0.0
    for labels,inputs in val_loader:
        inputs = inputs.to(device).float()
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            outputs = net(inputs)
            preds = outputs.argmax(axis=1)
            val_loss = criterion(outputs, labels)
            #val_losses.append(val_loss.item())
            #correct += pred.eq(target.data.view_as(pred)).sum()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
     
    epoch_loss = running_loss /len(val_loader.dataset)
    epoch_acc = running_corrects.double() / len(val_loader.dataset)
    val_losses.append(epoch_loss)
    val_accuracy.append(epoch_acc)
    print('Val Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    
    #chekpoint
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_model_wts = copy.deepcopy(net.state_dict())
    
    # early stopping, si el error aumenta más de 5 veces respecto al menor error,
    # terminamos el entrenamiento
    if epoch_loss > best_loss*5:
        print('\n'+'-' * 10+'Early Stopping'+'-' * 10+'\n')
        break

  print('Best val loss: {:.4f}'.format(best_loss))
  plt.figure()
  #2b. Graficar las curvas de loss de entrenamiento y validación
  plt.title("Error en cada epoca")
  plt.plot(train_counter, train_losses, label='Entrenamiento',color='blue')
  plt.plot(train_counter,val_losses, label='Validacion',color='red')
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.legend()
  plt.show()

  final = time.time()
  print('Training complete in {:.0f}m {:.0f}s'.format((final-inicio)//60, (final-inicio) % 60))

  net.load_state_dict(best_model_wts)
  return net

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools


def plot_confusion_matrix(cm, classes,accuracy,N,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):


  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  plt.figure(figsize=(10,7))
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title+'\n Accuracy = '+str(round(accuracy,2))+'%'+' N = '+str(N))
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()

# calculate accuracy
from sklearn.metrics import  accuracy_score

def evaluar_red(best_net,N,plot=True):
    #Evaluamos la red con los conjuntos de entrenamiento y validación
    best_net.eval()

    y_pred = []
    y_train = []
    for labels,inputs in train_loader:
        inputs = inputs.to(device).float()
        labels = labels.to(device)
        y_train += labels.cpu().tolist()
        with torch.no_grad():
            outputs = best_net(inputs)
            preds = outputs.argmax(axis=1)
            y_pred += preds.cpu().tolist()

    accuracy = accuracy_score(y_train, y_pred)*100
    cm = confusion_matrix(y_train, y_pred)
    if plot:
        plot_confusion_matrix(cm, list(range(10)), accuracy,N, title="Matriz de confusión Conjunto Entrenamiento")

    #Val
    best_net.eval()
    y_pred = []
    y_val = []
    for labels,inputs in val_loader:
        inputs = inputs.to(device).float()
        labels = labels.to(device)
        y_val += labels.cpu().tolist()
        with torch.no_grad():
            outputs = best_net(inputs)
            preds = outputs.argmax(axis=1)
            y_pred += preds.cpu().tolist()

    accuracy = accuracy_score(y_val, y_pred)*100        
    cm = confusion_matrix(y_val, y_pred)
    if plot:
        plot_confusion_matrix(cm, list(range(10)), accuracy,N, title="Matriz de confusión Conjunto Validación")
    #retornamos el accuracy en el conjunto de validación
    return accuracy

"""Inicializamos la red."""

N = 256
net = MyNet(N)
net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

"""## Sección 2a y 2b. Entrenar y graficar las curvas de loss de entrenamiento y validación"""

best_net_1 = train(net, optimizer, num_epocas=15)

"""# Sección 2.c Evaluar la red sobre los conjuntos de entrenamiento y validación, usando el mejor checkpoint almacenado"""

evaluar_red(best_net_1,N=256)

"""# Sección 3. Modificar el valor de N, repitiendo el Paso 2 hasta obtener una red con un buen desempeño.

Se prueban distintos valores de N, buscando obtener una red con mejor desempeño. Se selecciona el N que arroja mayor accuracy en el conjunto de validación.
"""

models = []
accuracys = []
n_list = [16,32,64,128,256]
for N in n_list:
    print(f'\nModel con N={N}')
    net = MyNet(N)
    net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3) 
    best_net = train(net, optimizer, num_epocas=20)
    models.append(best_net)
    accuracys.append(evaluar_red(best_net,N,plot=False))

"""## Resultados"""

plt.plot(n_list,accuracys,'o-')
plt.title('Accuracy validación vs N ')
plt.ylabel('Accuracy validación')
plt.xlabel('N (Neuronas última capa oculta)')
plt.grid()
# Texto en la gráfica en coordenadas (x,y)
idx = np.argmax(accuracys)

texto1 = plt.text( n_list[idx]+15,accuracys[idx]-0.05, f'N={n_list[idx]}, A='+str(round(accuracys[idx],2))+'%', fontsize=10)

N = n_list[idx]

#evaluamos el desempeño con el modelo que entrega mejor accuracy de validación
evaluar_red(models[idx],n_list[idx])

"""# Sección 4. Repetir el Paso 2 usando dos números distintos de capas convolucionales y elija el que genere los mejores resultados"""

# agregando una capa convolucional más
class MyNet2(nn.Module):
 def __init__(self, N=128):
    super(MyNet2, self).__init__()
    self.nclasses = 10
    #nn.Conv2d(in_channels, out_channels, kernel_size)
    self.conv1 = nn.Conv2d(3, 64, 3, padding = 1) #64 filtros de 3x3, 3 canales de entrada
    self.conv2 = nn.Conv2d(64, 64, 3, padding = 1) 
    self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
    self.conv4 = nn.Conv2d(128, 256, 3, padding = 1)

    self.bn1 = torch.nn.BatchNorm2d(64)
    self.bn2 = torch.nn.BatchNorm2d(64)
    self.bn3 = torch.nn.BatchNorm2d(128)
    self.bn4 = torch.nn.BatchNorm2d(256)

    self.MaxPool = nn.MaxPool2d(2, 2)

    self.fc1 = nn.Linear(4096, 1024)
    self.fc2 = nn.Linear(1024, 24*N)
    self.fc3 = nn.Linear(24*N, N)
    self.fc_last = nn.Linear(N, self.nclasses)
 def forward(self, x):
    x = self.bn1(F.relu(self.conv1(x)))
    x = self.MaxPool(self.bn2(F.relu(self.conv2(x))))
    x = self.MaxPool(self.bn3(F.relu(self.conv3(x))))
    x = self.MaxPool(self.bn4(F.relu(self.conv4(x))))

    #transformamos el tensor de una capa convolucional a una capa fully connected
    x = x.view(x.size()[0], -1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = self.fc_last(x)
    return x

print(f'\nModel con N={N}')
net3 = MyNet2(N)
net3.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net3.parameters(), lr=1e-3) 
best_net3 = train(net3, optimizer, num_epocas=20)

evaluar_red(best_net3,N)

#agregando 2 capas convolucionales más
class MyNet3(nn.Module):
 def __init__(self, N=16):
    super(MyNet3, self).__init__()
    self.nclasses = 10
    #nn.Conv2d(in_channels, out_channels, kernel_size)
    self.conv1 = nn.Conv2d(3, 64, 3, padding = 1) #64 filtros de 3x3, 3 canales de entrada
    self.conv2 = nn.Conv2d(64, 64, 3, padding = 1) 
    self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
    self.conv4 = nn.Conv2d(128, 256, 3, padding = 1)
    self.conv5 = nn.Conv2d(256, 512, 3, padding = 1)

    self.bn1 = torch.nn.BatchNorm2d(64)
    self.bn2 = torch.nn.BatchNorm2d(64)
    self.bn3 = torch.nn.BatchNorm2d(128)
    self.bn4 = torch.nn.BatchNorm2d(256)
    self.bn5 = torch.nn.BatchNorm2d(512)

    self.MaxPool = nn.MaxPool2d(2, 2)

    self.fc1 = nn.Linear(8192, 1024)
    self.fc2 = nn.Linear(1024, 24*N)
    self.fc3 = nn.Linear(24*N, N)
    self.fc_last = nn.Linear(N, self.nclasses)
 def forward(self, x):
    x = self.bn1(F.relu(self.conv1(x)))
    x = self.MaxPool(self.bn2(F.relu(self.conv2(x))))
    x = self.MaxPool(self.bn3(F.relu(self.conv3(x))))
    x = self.bn4(F.relu(self.conv4(x)))
    x = self.MaxPool(self.bn5(F.relu(self.conv5(x))))

    #transformamos el tensor de una capa convolucional a una capa fully connected
    x = x.view(x.size()[0], -1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = self.fc_last(x)
    return x

print(f'\nModel con N={N}')
net4 = MyNet3(N)
net4.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net4.parameters(), lr=1e-3) 
best_net4 = train(net4, optimizer, num_epocas=20)

evaluar_red(best_net4,N)

"""# Sección 5. Usando la mejor configuración obtenida en los pasos anteriores, evaluar la mejor red sobre el conjunto de prueba."""

# calculate accuracy
from sklearn.metrics import  accuracy_score

#Evaluamos la red con los conjuntos de prueba
best_net4.eval()
y_pred = []
y_test = []
for labels,inputs in test_loader:
    inputs = inputs.to(device).float()
    labels = labels.to(device)
    y_test += labels.cpu().tolist()
    with torch.no_grad():
        outputs = best_net4(inputs)
        preds = outputs.argmax(axis=1)
        y_pred += preds.cpu().tolist()

accuracy = accuracy_score(y_test, y_pred)*100        
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, list(range(10)), accuracy,N, title="Matriz de confusión Conjunto Prueba")