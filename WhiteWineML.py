''' changing the trainig size and testing size may change the success rate , for training = 3500 and testing = 1396 , sucess rate is 53.58% '''
''' email : montassar.naghmouchi@yahoo.com '''

from sklearn import cross_validation,metrics
from sklearn.linear_model import LogisticRegression
import pandas as pd

def data_extracting(filename):
  datafile=pd.read_table(filename,sep=";",header=1)
  data = datafile.as_matrix()
  return(data)
def parameters(data):
  return (data[:,0:11])
def results(data):
  return(data[:,11])
  
def main():
  data = data_extracting('WhiteWineData.txt')
  X=parameters(data)
  Y=results(data)
  #print(" training set ")
  #print(X)        
  X_app,X_test,y_app,y_test = cross_validation.train_test_split(X,Y,test_size =1396,train_size=3500,random_state=0)
  #print(X_app.shape,X_test.shape,y_app.shape,y_test.shape)
  Lr = LogisticRegression()
  #print(X_app) 
  try :
    model = Lr.fit(X_app,y_app)
    y_predict = model.predict(X_test)
    print('success rate = '+str(model.score(X_test,y_test,sample_weight=None))+'/1')
    print('Error Rate = '+str(1.0 - metrics.accuracy_score(y_test,y_predict))+'/1')
    print('success percentage = '+str(model.score(X_test,y_test,sample_weight=None)*100)+"%")
    cm = metrics.confusion_matrix(y_test,y_predict) 
    print('Confusion Matrix = ')
    print(cm) 
  except ValueError as error:
    print(error)
    probas = Lr.predict_proba(X_test)
    print(probas)
    score = probas[:,1]
    print(score)


if __name__=='__main__' :
    main()
 
