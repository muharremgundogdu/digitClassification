import numpy as np
import cv2
import os 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import pickle
import tensorflow as tf
# train_test_split ile veriyi train ve test olmak uzere ikiye ayiriyoruz
# gorsellestirmek icin seaborn ve matplotlib
# ImageDataGenerator ile farkli resimler generate edecegiz
# pickle i modeli yuklemek ve kaydetmek icin kullanacagiz

# verimiz myData klasorunde. 0 dan 9 a kadar rakamlar var hepsi ayni boyutta ama kalinlik ve fontlari farkli

path = "myData"

myList = os.listdir(path)
noOfClasses = len(myList)
print("Label(sinif) sayisi: ",noOfClasses)   # 10

images = []
classNo = []

for i in range(noOfClasses):
    myImageList = os.listdir(path + "\\" + str(i))
    for j in myImageList:
        img = cv2.imread(path + "\\" + str(i) + "\\" + j)  # resimleri aldik
        img = cv2.resize(img , (32,32))
        images.append(img)
        classNo.append(i)
        
print(len(images))
print(len(classNo))

images = np.array(images)
classNo = np.array(classNo)

print(images.shape)   # (10160, 32, 32, 3)  -> 10160 resim , 32x32 , 3 -> rgb
print(classNo.shape)  # (10160,)  -> vektor oldugundan 1

# veriyi ayir -> once train_test_split ile train ve testi ayir. sonra traini ayir ve train ve validationu elde et
x_train , x_test , y_train , y_test = train_test_split(images , classNo , test_size = 0.5 , random_state = 42)
x_train , x_validation , y_train , y_validation = train_test_split(x_train , y_train , test_size = 0.2 , random_state = 42)

print(images.shape)
print(x_train.shape)
print(x_test.shape)
print(x_validation.shape)

"""
# gorsellestirip bakalim
fig , axes = plt.subplots(3 , 1 , figsize = (7,7))
fig.subplots_adjust(hspace = 0.5)

sns.countplot(y_train , ax = axes[0])
axes[0].set_title("y_train")

sns.countplot(y_test , ax = axes[1])
axes[1].set_title("y_test")

sns.countplot(y_validation , ax = axes[2])
axes[2].set_title("y_validation")
"""

# preprocess asamasi
def preProcess(img):
    img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)  # histogramini 0 255 arasina kadar genislettik
    img = img / 255  # normalize
    
    return img

"""
# gorsellestirip bakalim
img = preProcess(x_train[50])
img = cv2.resize(img , (300,300))
cv2.imshow("Preprocess" , img)
"""

# bu preprocess islemini tum verimize uygulayacagiz -> map methodu ile
x_train = np.array(list(map(preProcess , x_train))) # map icerisine 2 input alir -> birincisi fonksiyon. bu fonksiyonu
# 2.parametrenin hepsine uygular. sonra bunu liste icine koyduk 
x_test = np.array(list(map(preProcess , x_test)))
x_validation = np.array(list(map(preProcess , x_validation)))

# veriyi reshape yap -> veriyi egitime hazir hale getir
x_train = x_train.reshape(-1 , 32 , 32 , 1)  # -1 -> otomatik olarak hesaplayacak
print(x_train.shape)
x_test = x_test.reshape(-1 , 32 , 32 , 1)
x_validation = x_validation.reshape(-1 , 32 , 32 , 1)

# data generate
dataGen = ImageDataGenerator(width_shift_range = 0.1 , height_shift_range = 0.1 , zoom_range = 0.1 , rotation_range = 10)
# width_shift_range = 0.1 -> 0.1 oraninda genislikte kayiyor
dataGen.fit(x_train)

# y_train , y_test , y_validation u kategorical hale getir. kategorical hale  getirmek onehotencoder ile 
# ayni gorev oluyor. bunu keras icin yapmamiz gerekir
y_train = to_categorical(y_train , noOfClasses)
y_test = to_categorical(y_test , noOfClasses)
y_validation = to_categorical(y_validation , noOfClasses)

# modeli insa et
model = Sequential()  # sequential bir temel olusturduk
model.add(Conv2D(input_shape = (32,32,1) , filters = 8 , kernel_size = (5,5) , activation = "relu" , padding = "same"))
# bu sequential temelin uzerine ekliyoruz
# same padding -> bir sira piksel ekliyor

# max pooling layer ekle
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(filters = 16 , kernel_size = (3,3) , activation = "relu" , padding = "same"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Dropout(0.2))  # yeni veri urettigimiz icin ezberlemeyi engellememiz gerekiyor bu yuzden dropout ekliyoruz
model.add(Flatten())
model.add(Dense(units = 256 , activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(units = noOfClasses , activation = "softmax"))  # cikti -> output layer

# modeli compile etmek icin gerekli olan parametreleri yaz
model.compile(loss = "categorical_crossentropy" , optimizer = ("Adam") , metrics = ["accuracy"])

batch_size = 250

# modeli egit
hist = model.fit_generator(dataGen.flow(x_train , y_train , batch_size = batch_size) , 
                                        validation_data = (x_validation , y_validation) , epochs = 15 ,
                                        steps_per_epoch = x_train.shape[0]//batch_size , shuffle = 1)
# history diye parametre belirliyoruz bu sayede modelimizin ciktilarini gorsellestirebilecegiz
# step_per_epoch -> her bir epochdaki egitimimiz. batch size ile kalansiz bolumune baktik
# shuffle -> veriyi karistiriyor

model.save("./model_trained_new.h5")

# %% degerlendirme
hist.history.keys()
plt.figure()
plt.plot(hist.history["loss"] , label = "Egitim Loss")
plt.plot(hist.history["val_loss"] , label = "Val Loss")
plt.legend()
plt.show

plt.figure()
plt.plot(hist.history["accuracy"] , label = "Egitim Accuracy")
plt.plot(hist.history["val_accuracy"] , label = "Val Accuracy")
plt.legend()
plt.show

# testin sonuclari
score = model.evaluate(x_test , y_test , verbose = 1)
print("Test Loss: " , score[0]) # score nin 0.indeksinde loss vardir
print("Accuracy Loss: " , score[1]) # score nin 1.indeksinde accuracy vardir
"""
model.evaluate(): Bu, modelin bir test seti uzerinde degerlendirilmesini saglar. Bu metod, 
test veri kumesini (genellikle x_test olarak adlandirilir), test etiketlerini (y_test olarak adlandirilir) 
ve opsiyonel olarak birkaç ek argumani alir.

verbose=1: Bu, çıktının ayrintilı olmasini saglayan bir parametredir. 1 ayrintili ciktilari temsil eder, 
0 ise sessiz modu temsil eder. Bu, modelin degerlendirme islemi sirasinda ilerleme cubugu gibi ek bilgilerin 
goruntulenip goruntulenmeyecegini kontrol eder.
"""

# tum classlarin sonuclari  ->  confusion matrix ile
y_pred = model.predict(x_validation)

y_pred_class = np.argmax(y_pred , axis = 1)

Y_true = np.argmax(y_validation , axis = 1)

cm = confusion_matrix(Y_true , y_pred_class)

f , ax = plt.subplots(figsize = (8,8))

sns.heatmap(cm , annot = True , linewidths = 0.01 , cmap = "Greens" , linecolor = "gray" , fmt = ".1f" , ax = ax)
# linewidth -> aradaki cizgilerin kalinligi , linecolor -> aradaki cizginin kalinligi , .1f virgulden sonraki basamak
# annot=true -> heatmap in uzerinde sayilar yazacak
plt.xlabel("predicted")
plt.ylabel("true")
plt.title("confusion matrix")
plt.show()
# grafik dogru cikmadi
# bu kod satirinda x ekseninde tahminler , y ekseninde o tahminin gercek degeri bulunuyor. gercek deger -> Y_truth
# ama grafik yanlis cikti


