import cv2
import numpy as np
from keras.models import Sequential, load_model

# resmi yine preprocess edecegiz cunku kameradan aldigimiz verimizi preprocess etmeden noral networke sokarsak calismaz
# preprocess islemi
def preProcess(img):
    img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)  # histogramini 0 255 arasina kadar genislettik
    img = img / 255  # normalize
    return img

cap = cv2.VideoCapture(0)
cap.set(3 , 480)
cap.set(4 , 480)

model = load_model("model_trained_new.h5")
# egitmis oldugumuz modeli iceriye yukledik.

while True:
    success , frame = cap.read()
    
    img = np.asarray(frame)  # frame yi arraye cevirdik
    img = cv2.resize(img,(32,32))  # noral networku egitirken inputun shape'ini 32 32 almistik o yuzden burayi da oyle yapiyoruz
    img = preProcess(img)
    
    img = img.reshape(1 , 32 , 32 , 1)  # ilk 1 -> 1 resim oldugunu , 32 ye 32 boyut , sondaki 1 channel -> siyah beyaz
    
    # predict islemi
    predictions = model.predict(img)    
    classIndex = np.argmax(predictions)
    
    probVal = np.amax(predictions)  # olasilik
    print(classIndex , probVal)
    
    if probVal > 0.7:
        cv2.putText(frame , str(classIndex) + "   " + str(probVal) , (50,50) , cv2.FONT_HERSHEY_DUPLEX , 1 , (0,255,0) , 1)
    
    cv2.imshow("Rakam Siniflandirma" , frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()




