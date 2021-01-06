import nltk
import pandas as pd
import re 
import glob
import math
import sys
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression



def feature_matrix(ham,spam):
    #"C:\\Users\\Parashar Parikh\\Desktop\\UTD Sem1\\Machie Learning\\hw2\\hw2_train\\train\\ham\\*.txt"
    words=pd.DataFrame()
    final_data_ham=pd.DataFrame()
    final_data_spam=pd.DataFrame()
    final=pd.DataFrame()
    final1=pd.DataFrame()
    final2=pd.DataFrame()
    final_bow=pd.DataFrame()
    final1_bow=pd.DataFrame()
    final2_bow=pd.DataFrame()
    for filename in glob.glob(ham):
        #filename="C:\\Users\\Parashar Parikh\\Desktop\\UTD Sem1\\Machie Learning\\hw2\\hw2_train\\train\\spam\\0170.2004-01-09.GP.spam.txt"
        data = open(filename ,"r").read() 
        corpus = nltk.sent_tokenize(data)
        for i in range(len(corpus)):
                corpus [i] = corpus [i].lower()
                corpus [i] = re.sub(r'\W',' ',corpus [i])
                #corpus [i] = re.sub(r'\s+',' ',corpus [i])
                #corpus [i] = re.sub(r'(^|\W)\d+','',corpus [i])
                #corpus [i] = re.sub(r'\W*\b\w{1}\b',' ', corpus [i])
        #print(corpus)

        wordfreq = {}
        for sentence in corpus:
           tokens = nltk.word_tokenize(sentence)
           for token in tokens:
                    if token not in wordfreq.keys():
                        wordfreq[token] = 1
                    else:
                        wordfreq[token] += 1

        final_data_ham=final_data_ham.append(pd.DataFrame.from_dict(list(wordfreq.items())))

    final_data_ham.columns=['words','freq']
    #final_data_ham=final_data_ham[final_data_ham['freq']>1]
    words_ham=final_data_ham.words.unique()
    words_ham=pd.DataFrame(words_ham)

    for filename in glob.glob(spam):
        data = open(filename,"r",encoding='utf-8',errors='ignore').read() 
        corpus = nltk.sent_tokenize(data)
        for i in range(len(corpus)):
            corpus [i] = corpus [i].lower()
            corpus [i] = re.sub(r'\W',' ',corpus [i])
            #corpus [i] = re.sub(r'\s+',' ',corpus [i])
            #corpus [i] = re.sub(r'(^|\W)\d+','',corpus [i])
            #corpus [i] = re.sub(r'\W*\b\w{1}\b',' ', corpus [i])
        #print(corpus)

        wordfreq = {}
        for sentence in corpus:
            tokens = nltk.word_tokenize(sentence)
            for token in tokens:
                if token not in wordfreq.keys():
                    wordfreq[token] = 1
                else:
                    wordfreq[token] += 1



        final_data_spam=final_data_spam.append(pd.DataFrame.from_dict(list(wordfreq.items())))

    final_data_spam.columns=['words','freq']
    #final_data_spam=final_data_spam[final_data_spam['freq']>1]
    words_spam=final_data_spam.words.unique()
    words_spam=pd.DataFrame(words_spam)


    words=pd.concat([words_ham,words_spam],ignore_index=True)
    words=words.transpose()
    words.columns = words.iloc[0]
    #words.to_dict(l)




    #generating matrix bernoulli model



    for filename in glob.glob(ham):
        #filename="C:\\Users\\Parashar Parikh\\Desktop\\UTD Sem1\\Machie Learning\\hw2\\hw2_train\\train\\ham\\0004.1999-12-14.farmer.ham.txt"
        data = open(filename,"r",encoding='utf-8',errors='ignore').read() 
        corpus = nltk.word_tokenize(data)


        freq = {el:0 for el in words.columns.unique()}
        for sentence in corpus:
                for i in freq.keys():
                    if sentence.lower()== i :
                        freq[i]= 1
        freq1 = {el:0 for el in words.columns.unique()}
        for sentence in corpus:
            #tokens = nltk.word_tokenize(sentence)
            #for token in sentence:
                for i in freq1.keys():
                    if sentence.lower()== i :
                        #print('*')
                        freq1[i]+=1               

        temp=pd.DataFrame.from_dict(list(freq.items()))
        temp=temp.transpose()     
        final1=final1.append(temp.iloc[1],ignore_index=True)
        temp1=pd.DataFrame.from_dict(list(freq1.items()))
        temp1=temp1.transpose()     
        final1_bow=final1_bow.append(temp1.iloc[1],ignore_index=True)
    final1_bow.columns=words.columns.unique()
    final1_bow['target']='ham'
    final1.columns=words.columns.unique()
    final1['target']='ham'



    for filename in glob.glob(spam):
        #filename="C:\\Users\\Parashar Parikh\\Desktop\\UTD Sem1\\Machie Learning\\hw2\\hw2_train\\train\\ham\\0004.1999-12-14.farmer.ham.txt"
        data = open(filename,"r",encoding='utf-8',errors='ignore').read() 
        corpus = nltk.word_tokenize(data)

        freq = {el:0 for el in words.columns.unique()}
        freq1 = {el:0 for el in words.columns.unique()}
        
        #wordfreq.keys=words.columns
        #final.columns=freq.keys()
        for sentence in corpus:
            #tokens = nltk.word_tokenize(sentence)
            #for token in sentence:
                for i in freq.keys():
                    if sentence.lower()== i :
                        #print('*')
                        freq[i]= 1
                        freq1[i]+=1 
                              

        temp=pd.DataFrame.from_dict(list(freq.items()))
        temp=temp.transpose()     
        final2=final2.append(temp.iloc[1],ignore_index=True)
        temp1=pd.DataFrame.from_dict(list(freq1.items()))
        temp1=temp1.transpose()     
        final2_bow=final2_bow.append(temp1.iloc[1],ignore_index=True)
    final2_bow.columns=words.columns.unique()
    final2_bow['target']='spam'
    final2.columns=words.columns.unique()
    final2['target']='spam'   
    final=pd.concat([final1,final2],ignore_index=True) 
    final_bow=pd.concat([final1_bow,final2_bow],ignore_index=True)         

    return final,final_bow

def multi_naive(final_bow):
    total_ham = final_bow['target'].value_counts()[0]
    total_spam= final_bow['target'].value_counts()[1]
    # prior
    probability_ham = total_ham / (total_ham + total_spam)
    probability_spam = total_spam / (total_ham + total_spam)
    prior = [math.log(probability_spam),math.log( probability_ham)]
    
    v=final_bow.shape[1]
    temp_ham=final_bow[final_bow['target']=='ham']
    temp_spam=final_bow[final_bow['target']=='spam']
    
    sum1=pd.DataFrame(temp_ham.sum(axis=0))
    total_ham=0
    for i in range(len(sum1)-1):
        total_ham=total_ham+sum1.iloc[i]
    sum2=pd.DataFrame(temp_spam.sum(axis=0))
    total_spam=0
    for i in range(len(sum2)-1):
        total_spam=total_spam+sum2.iloc[i]
        
    prob=pd.DataFrame(columns=final_bow.columns)
    prob=prob.drop(['target'], axis=1)
    
    condition_prob_spam=[]
    condition_prob_ham=[]
    logcondition_prob_spam=[]
    logcondition_prob_ham=[]
    
    for i in range(prob.shape[1]):
        condition_prob_ham.append((sum1.iloc[i]+1)/(total_ham[0]+(v-1)))
        condition_prob_spam.append((sum2.iloc[i]+1)/(total_spam[0]+(v-1)))
        logcondition_prob_ham.append(math.log((sum1.iloc[i]+1)/(total_ham[0]+(v-1))))
        logcondition_prob_spam.append(math.log((sum2.iloc[i]+1)/(total_spam[0]+(v-1)))) 
    logcondition_prob_ham=pd.DataFrame(logcondition_prob_ham).transpose()
    logcondition_prob_spam=pd.DataFrame(logcondition_prob_spam).transpose()
    frames=[logcondition_prob_ham,logcondition_prob_spam]
    prob=pd.concat(frames,ignore_index=True)
    #prob.columns=final_bow.columns
    prob['target']=['ham','spam']
    prob.columns=final_bow.columns    
    return prob, prior

def discrete_naive(final):
    
    total_ham = final_bow['target'].value_counts()[0]
    total_spam= final_bow['target'].value_counts()[1]
    # prior
    probability_ham = total_ham / (total_ham + total_spam)
    probability_spam = total_spam / (total_ham + total_spam)
    prior = [math.log(probability_spam),math.log( probability_ham)]
    
    
    temp_ham=final[final['target']=='ham']
    temp_spam=final[final['target']=='spam']
    
    sum1=pd.DataFrame(temp_ham.sum(axis=0))
    total_ham=0
    for i in range(len(sum1)-1):
        total_ham=total_ham+sum1.iloc[i]
    sum2=pd.DataFrame(temp_spam.sum(axis=0))
    total_spam=0
    for i in range(len(sum2)-1):
        total_spam=total_spam+sum2.iloc[i]
    total_ham = final_bow['target'].value_counts()[0]
    total_spam= final_bow['target'].value_counts()[1]    
    prob_bernoulli=pd.DataFrame(columns=final.columns)
    prob_bernoulli=prob_bernoulli.drop(['target'], axis=1)
    condition_prob_spam=[]
    condition_prob_ham=[]
    logcondition_prob_spam=[]
    logcondition_prob_ham=[]
    
    for i in range(prob_bernoulli.shape[1]):
        condition_prob_ham.append((sum1.iloc[i]+1)/(2+total_ham))
        condition_prob_spam.append((sum2.iloc[i]+1)/(2+total_spam))
        logcondition_prob_ham.append(math.log((sum1.iloc[i]+1)/(2+total_ham)))
        logcondition_prob_spam.append(math.log((sum2.iloc[i]+1)/(2+total_spam))) 
    logcondition_prob_ham=pd.DataFrame(logcondition_prob_ham).transpose()
    logcondition_prob_spam=pd.DataFrame(logcondition_prob_spam).transpose()
    frames=[logcondition_prob_ham,logcondition_prob_spam]
    prob_bernoulli=pd.concat(frames,ignore_index=True)
    
    #prob.columns=final_bow.columns
    prob_bernoulli['target']=['ham','spam']    
    prob_bernoulli.columns=final.columns 
    return prob_bernoulli,prior

def find_result(prior,prob,ham_test,spam_test):
    prob=prob.drop(['target'], axis=1)
    original=[]
    counter_ham=0
    nf_ham=0
    prediction=[]
   
    for filename in glob.glob(ham_test):
        #filename="C:\\Users\\Parashar Parikh\\Desktop\\UTD Sem1\\Machie Learning\\hw2\\hw2_test\\test\\ham\\0003.1999-12-14.farmer.ham.txt"
        original.append(0)
        nf_ham+=1
        data = open(filename,"r",encoding='utf-8',errors='ignore').read() 
        corpus = nltk.word_tokenize(data)
        score = [0, 0]
        for word in corpus:
            score[0] = prior[0]
            score[1] = prior[1] 
            for i in prob.columns:
                if word.lower()== i :
                        score[0] += prob.loc[0][i]
                        score[1] += prob.loc[1][i]
                        
    
        if score[0] > score[1]:
            counter_ham+=1
            prediction.append(0)
        else:
            prediction.append(1)

        
    nf_spam=0
    counter_spam=0
    for filename in glob.glob(spam_test):
        #filename="C:\\Users\\Parashar Parikh\\Desktop\\UTD Sem1\\Machie Learning\\hw2\\hw2_test\\test\\ham\\0003.1999-12-14.farmer.ham.txt"
        original.append(1)
        nf_spam+=1
        data = open(filename,"r",encoding='utf-8',errors='ignore').read() 
        corpus = nltk.word_tokenize(data)
        score = [0, 0]
        for word in corpus:
            score[0] = prior[0]
            score[1] = prior[1] 
            for i in prob.columns:
                if word.lower()== i :
                        score[0] += prob.loc[0][i]
                        score[1] += prob.loc[1][i]
                        
    
        if score[0] < score[1]:
            counter_spam+=1
            prediction.append(1)
        else:
            prediction.append(0)
            
    #print(original)
    #print(prediction) 
    #original=pd.DataFrame.from_dict(list(original))
    #prediction=pd.DataFrame.from_dict(list(prediction))
    accuracy = accuracy_score(original, prediction)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(original, prediction)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(original, prediction)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(original, prediction)
    print('F1 score: %f' % f1)
    accuracy1=(counter_ham+counter_spam)/(nf_ham+nf_spam)
    return accuracy1,original,prediction    
    
def sgd(final,final_bow,ham_test,spam_test):
    
    x_train=final.loc[: , final.columns!='target']
    x_train_bow=final_bow.loc[: , final_bow.columns!='target']
    
    final['target'] = pd.factorize(final.target)[0] 
    y_train=final['target']
    final_bow['target'] = pd.factorize(final_bow.target)[0] 
    y_train_bow=final_bow['target']
    
    #x_test=pd.DataFrame()
    #x_test_bow=pd.DataFrame()
    final1=pd.DataFrame()
    final1_bow=pd.DataFrame()
    final2=pd.DataFrame()
    final2_bow=pd.DataFrame()
    
    original=[]
    #print("x train",x_train)
    #print('columns',x_train.columns)
    for filename in glob.glob(ham_test):
        original.append(0)
        data = open(filename,"r",encoding='utf-8',errors='ignore').read() 
        corpus = nltk.word_tokenize(data)
        #print("corpus",corpus)
        freq = {el:0 for el in x_train.columns}
        #print("Keys",freq.keys())
        for sentence in corpus:
            for i in freq.keys():
                if sentence.lower()== i :
                    freq[i]= 1
        freq1 = {el:0 for el in x_train.columns}
        for sentence in corpus:
            for i in freq1.keys():
                if sentence.lower()== i :
                    freq1[i]+=1                

        temp=pd.DataFrame.from_dict(list(freq.items()))
        #print(temp)
        temp=temp.transpose()     
        final1=final1.append(temp.iloc[1],ignore_index=True)
        temp1=pd.DataFrame.from_dict(list(freq1.items()))
        temp1=temp1.transpose()     
        final1_bow=final1_bow.append(temp1.iloc[1],ignore_index=True)
    final1_bow.columns=x_train_bow.columns
    final1.columns=x_train_bow.columns
   



    for filename in glob.glob(spam_test):
        #filename="C:\\Users\\Parashar Parikh\\Desktop\\UTD Sem1\\Machie Learning\\hw2\\hw2_train\\train\\ham\\0004.1999-12-14.farmer.ham.txt"
        original.append(1)
        data = open(filename,"r",encoding='utf-8',errors='ignore').read() 
        corpus = nltk.word_tokenize(data)

        freq_spam = {el:0 for el in x_train.columns}
            #wordfreq.keys=words.columns
            #final.columns=freq.keys()
        for sentence in corpus:
            #tokens = nltk.word_tokenize(sentence)
            #for token in sentence:
                for i in freq_spam.keys():
                    if sentence.lower()== i :
                        freq_spam[i]= 1
        freq1_spam = {el:0 for el in x_train.columns}
            #wordfreq.keys=words.columns
            #final.columns=freq.keys()
        for sentence in corpus:
            #tokens = nltk.word_tokenize(sentence)
            #for token in sentence:
                for i in freq1_spam.keys():
                    if sentence.lower()== i :
                        freq1_spam[i]+=1

        temp=pd.DataFrame.from_dict(list(freq_spam.items()))
        temp=temp.transpose()     
        final2=final2.append(temp.iloc[1],ignore_index=True)
        temp1=pd.DataFrame.from_dict(list(freq1_spam.items()))
        temp1=temp1.transpose()     
        final2_bow=final2_bow.append(temp1.iloc[1],ignore_index=True)
    final2_bow.columns=x_train_bow.columns
    final2.columns=x_train_bow.columns
        
    #print("final1",final1)
    #print("final2",final2)
    final_test=pd.concat([final1,final2],ignore_index=True) 
    final_bow_test=pd.concat([final1_bow,final2_bow],ignore_index=True)
    x_test=final_test
    x_test_bow=final_bow_test
    
    #print("x test",x_test)
    #print("x test bow",x_test_bow)
    
    y_test=pd.DataFrame.from_dict(list(original))
    #print(y_test)
    print("logistic regression results:")
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    print('\nAccuracy:', accuracy_score(y_test, y_pred))
    print('F1 score:', f1_score(y_test, y_pred))
    print('Precision score', precision_score(y_test, y_pred))
    print('recall ', recall_score(y_test, y_pred))
    
    print("Results using  sgd model")
    
    parameter_grid= {
                        'loss':["squared_hinge","hinge"],
                        'penalty':["l1","l2","elasticnet"],
                        'max_iter':[50,70,90,100],
                        'alpha':[0.0001,0.001,0.01,0.1,1,10,100]
                        }
    clf = SGDClassifier(random_state=0, class_weight='balanced',
                    loss='log', penalty='elasticnet')
    clf = GridSearchCV(SGDClassifier(), parameter_grid)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    print("\nResults of bernoullie sgd model")
    print('Accuracy:', accuracy_score(y_test, pred))
    print('F1 score:', f1_score(y_test, pred))
    print('Precision score', precision_score(y_test, pred))
    print('recall ', recall_score(y_test, pred))
    print("Best parameters")
    print(clf.best_params_)
    print(clf.best_score_)
    
    clf.fit(x_train_bow, y_train_bow)
    pred_bow = clf.predict(x_test_bow)
    print("Result of bag of words sgd model")
    print('\nAccuracy:', accuracy_score(y_test, pred_bow))
    print('F1 score:', f1_score(y_test, pred_bow))
    print('Precision score', precision_score(y_test, pred_bow))
    print('recall ', recall_score(y_test, pred_bow))
    print("Best parameters")
    print(clf.best_params_)
    #print(clf.best_score_)    
    
    
    return pred


final=pd.DataFrame()
final_bow=pd.DataFrame()
prior=[]
prob=[]


if __name__ == '__main__':
    #ham="C:\\Users\\Parashar Parikh\\Desktop\\UTD Sem1\\Machie Learning\\hw2\\hw2_train\\train\\ham\\*.txt"
    #spam="C:\\Users\\Parashar Parikh\\Desktop\\UTD Sem1\\Machie Learning\\hw2\\hw2_train\\train\\spam\\*.txt"
    #ham_test="C:\\Users\\Parashar Parikh\\Desktop\\UTD Sem1\\Machie Learning\\hw2\\hw2_test\\test\\ham\\*.txt"
    #spam_test="C:\\Users\\Parashar Parikh\\Desktop\\UTD Sem1\\Machie Learning\\hw2\\hw2_test\\test\\spam\\*.txt"
    ham=sys.argv[1]
    spam=sys.argv[2]
    ham_test=sys.argv[3]
    spam_test=sys.argv[4]
    final,final_bow=feature_matrix(ham,spam)
    print("bag of words model")
    print(final_bow)
    print("Bernoullie model")
    print(final)
    
    prob,prior=multi_naive(final_bow)
    print("probebility using multinomial naive bayes model")
    print(prob)
    print("Results using multinomial naive bayes model")
    accuracy,original,predictoin=find_result(prior,prob,ham_test,spam_test)
    #print("accuracy"+ accuracy)
    prob_bernoulli,prior=discrete_naive(final)
    print("probebility using discrete naive bayes model")
    print(prob_bernoulli)
    print("Results using discreate naive bayes model")
    accuracy1,original1,predictoin1=find_result(prior,prob_bernoulli,ham_test,spam_test)
    
    pred=sgd(final,final_bow,ham_test,spam_test)
    
    
   # >python Hw2_cmd.py "C:\\Users\\Parashar Parikh\\Desktop\\UTD Sem1\\Machie Learning\\hw2\\hw2_train\\train\\ham\\*.txt" "C:\\Users\\Parashar Parikh\\Desktop\\UTD Sem1\\Machie Learning\\hw2\\hw2_train\\train\\spam\\*.txt" "C:\\Users\\Parashar Parikh\\Desktop\\UTD Sem1\\Machie Learning\\hw2\\hw2_test\\test\\ham\\*.txt" "C:\\Users\\Parashar Parikh\\Desktop\\UTD Sem1\\Machie Learning\\hw2\\hw2_test\\test\\spam\\*.txt"
    
    
    