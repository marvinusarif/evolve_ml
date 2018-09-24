from nlp.example_tfidf_lsi.generate_data import load_data
from nlp.vectorizer.dictionary_builder import load_dictionary
from nlp.vectorizer.tfidf_vectorizer import load_tfidf_model
from nlp.vectorizer.lsi_vectorizer import load_lsi_model, convert_text_to_lsi

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def labelling_sentiment(documents):
    labels=[]
    sizedocument=len(documents)
    for i in range(sizedocument):
        if (i % 2)==0:
            labels.append(1.0)
        else:
            labels.append(-1.0)

    return labels

def generate_lsi_vector(documents, lsi_model, dictionary, tfidf_model):
    list_of_lsi_vector=[]
    for d in documents:
        lsi_vector=convert_text_to_lsi(d, dictionary,
                                       tfidf_model, lsi_model)
        list_of_lsi_vector.append(lsi_vector)

    return list_of_lsi_vector


def get_lsi_values_only(list_of_lsi_vector):
    list_of_lsi_values=[]
    for lsivector in list_of_lsi_vector:
        lsivalues=[]
        for lv in lsivector:
            lsivalues.append(lv[1])
        list_of_lsi_values.append(lsivalues)

    return list_of_lsi_values


documents=load_data()
labels=labelling_sentiment(documents)

file_dictionary = 'artikel.dict'
file_tfidf_model = 'artikel.tfidf'
file_lsi_model = 'artikel.lsi'
dictionary=load_dictionary(file_dictionary)
tfidf_model=load_tfidf_model(file_tfidf_model)
lsi_model=load_lsi_model(file_lsi_model)

list_of_lsi_vector=generate_lsi_vector(documents, lsi_model,
                                       dictionary, tfidf_model)

list_of_lsi_values=get_lsi_values_only(list_of_lsi_vector)

print('text document label', labels[0])
print(documents[0])
print(list_of_lsi_values[0])

print('===================================')

print('text document label', labels[1])
print(documents[1])
print(list_of_lsi_values[1])

x_train=list_of_lsi_values[:2000]
x_test=list_of_lsi_values[2000:]

y_train=labels[:2000]
y_test=labels[2000:]

print('===============================================')
print(x_train[0])
print(y_train[0])

rfmodel=RandomForestClassifier(criterion='gini')
rfmodel.fit(x_train,y_train)

pred_x_train=rfmodel.predict(x_train)
pred_x_test=rfmodel.predict(x_test)

acc_train=accuracy_score(y_train, pred_x_train)
acc_test=accuracy_score(y_test, pred_x_test)

print('acc_train', acc_train)
print('acc_test', acc_test)
