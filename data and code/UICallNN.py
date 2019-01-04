import tkinter as tk 
from tkinter import *
import tkinter
from tkinter import messagebox
import tensorflow as tf
import numpy as np
import pandas as pd
import matlab.engine

window = Tk()
window.title("Welcome to Students Performance Prediction!")
 
window.geometry('680x500')
window.configure(bg='#D3ECF5')

'''testData = pd.read_excel('testData.xlsx')
#NN Hardcoded
def predictNN(A1, A2, A3, Q1, Q2):
    data = np.stack([testData.A1.values, testData.A2.values,
                     testData.A3.values,testData.Q1.values,testData.Q2.values,testData.GradeNN.values]).T
   
    
    for i in range(3):
        if (A1 == str(data[i][0]) and A2 == str(data[i][1]) and A3 == str(data[i][2]) 
            and Q1 == str(data[i][3]) and Q2 == str(data[i][4])):
            if (data[i][5] == 0):
                labelMessage.config(text="'Below Average' grade predicted, you must work harder!", font=("", 18), 
                                    fg='Red')

            elif (data[i][5] == 1):
                labelMessage.config(text="     'Average' grade predicted, work harder!", font=("", 18), 
                                    fg='#F58D25')
            elif (data[i][5] == 2):
                
                labelMessage.config(text="'Excellent' grade predicted, keep up the good work!", font=("", 18), 
                                    fg='Green')

#ANFIS hardcoded
def Anfis(a1,a2,a3,q1,q2):
    d = np.stack([testData.A1.values,testData.A2.values,testData.A3.values,testData.Q1.values,testData.Q2.values,testData.GradeANFIS.values]).T
    for i in range(3):
        if(float(a1)== d[i][0] and float(a2)== d[i][1] and float(a3)== d[i][2] 
        and float(q1)== d[i][3] and float(q2)== d[i][4]):
            if(d[i][5]<= 60.9):
                labelMessage.config(text="Predicted Score: "+str(d[i][5])+" Grade: below average, must work harder!",font=("",18),fg='red')
            elif(d[i][5]>= 61.0 and d[i][5]<= 79.9 ):
                labelMessage.config(text="Predicted Score: "+str(d[i][5])+" Grade: average, work harder",font=("",18),fg='#F58D25')
            else:
                labelMessage.config(text="Predicted Score: "+str(d[i][5])+" Grade: Excellent , keep up the good work!",font=("",18),fg='Green')
 # NN hardcoded           
def button1Listener():
    A1 = text_A1.get()
    A2 = text_A2.get()
    A3 = text_A3.get()
    Q1 = text_Q1.get()
    Q2 = text_Q2.get()
    
    
    if (A1 == "88" and A2 == "100" and A3 == "98" and
        Q1 == "80" and Q2 == "94"):
        predictNN(A1, A2, A3, Q1, Q2)     
    elif (A1 == "82" and A2 == "60" and A3 == "94" and
        Q1 == "50" and Q2 == "72"):
        predictNN(A1, A2, A3, Q1, Q2)
    elif (A1 == "77" and A2 == "76" and A3 == "46" and
        Q1 == "80" and Q2 == "62"):
        predictNN(A1, A2, A3, Q1, Q2)
    else:
        labelMessage.config(text="Data entered is not correct!", font=("", 20))


#ANFIS hardcoded       
def button2Listener():
    A1 = text_A1.get()
    A2 = text_A2.get()
    A3 = text_A3.get()
    Q1 = text_Q1.get()
    Q2 = text_Q2.get()
    
    #messagebox.showinfo("Title",""+A1+""+Q1)
    if (A1 == "88" and A2 == "100" and A3 == "98" and Q1 == "80" and Q2 == "94"):
        Anfis(A1, A2, A3, Q1, Q2)     

    elif (A1 == "82" and A2 == "60" and A3 == "94" and Q1 == "50" and Q2 == "72"):
        Anfis(A1, A2, A3, Q1, Q2)

    elif (A1 == "77" and A2 == "76" and A3 == "46" and Q1 == "80" and Q2 == "62"):
        Anfis(A1, A2, A3, Q1, Q2)

    else:
        labelMessage.config(text="Data entered is not correct!", font=("", 20))'''
 #-------------------------------------------- NN model ---------------------------------------------------    
def addLayer(inputs,in_size,out_size,activation_function=None):
        w = tf.Variable(tf.random_normal([in_size,out_size]))
   
        b = tf.Variable(tf.zeros([1,out_size])+0.1)
        f = tf.matmul(inputs,w) + b
    
        if activation_function is None:
            outputs = f
        else:
                outputs = activation_function(f)
        return outputs
def button3Listener():
    excelData = pd.read_excel('train.xlsx', dtype={'A1':float, 'A2':float, 'A3':float, 
                                              'Q1':float, 'Q2':float, 'Total':float,
                                              'c1':float, 'c2':float, 'c3':float})
    del(excelData['Total'])  
    
    #test data from user input
    A1 = text_A1.get()
    A2 = text_A2.get()
    A3 = text_A3.get()
    Q1 = text_Q1.get()
    Q2 = text_Q2.get()
    
    testdata = [[]]
    
    testdata[0] = np.stack([A1, A2, A3, Q1, Q2]).T
    
    target = np.stack([excelData.c1.values, excelData.c2.values, excelData.c3.values]).T
    
    data = np.stack([excelData.A1.values, excelData.A2.values,
                     excelData.A3.values,excelData.Q1.values,excelData.Q2.values]).T

    X = tf.placeholder(tf.float32,[None,5])
    y = tf.placeholder(tf.float32,[None,3])

    layer1 = addLayer(X,5,128,activation_function=tf.nn.sigmoid)
    layer2 = addLayer(layer1,128,128,activation_function=tf.nn.sigmoid)
    layer3 = addLayer(layer2,128,128,activation_function=tf.nn.sigmoid)
    layer4 = addLayer(layer3,128,128,activation_function=tf.nn.sigmoid)
    layer5 = addLayer(layer4,128,128,activation_function=tf.nn.sigmoid)
    
    pred = addLayer(layer5,128,3,activation_function=tf.nn.sigmoid)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=pred))

    correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))  #show index of the max value
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

    train_step = tf.train.AdamOptimizer(0.0005).minimize(loss)
    sess = tf.Session()
    #initialize the variables
    
    do_train = 0   #use the trained model if it is 0, set to 1 if want to train 
    saver = tf.train.Saver()
    if do_train == 1:
        
        sess.run(tf.global_variables_initializer())
        
        for i in range(100000):
    
            index = np.random.permutation(len(target))
            data = data[index]
            target = target[index]
            
            sess.run(train_step,feed_dict={X:data,y:target})
    
            if i%1000==0:
                #print(sess.run((loss,accuracy),feed_dict={X:data,y:target}))
                print(sess.run((tf.argmax(pred,1),accuracy),feed_dict={X:data,y:target}))
                #print(sess.run((loss,accuracy),feed_dict={X:dataT,y:targetT}))
                #print(sess.run(tf.argmax(pred,1),feed_dict={X:dataT,y:targetT}))
                #print(sess.run(pred,feed_dict={X:testData}))
        save_path = saver.save(sess, "saved_nn/save_net.ckpt")
        print("Save to path: ", save_path)
        print(sess.run(tf.argmax(pred,1),feed_dict={X:testdata}))
    else:
        tf.reset_default_graph()
        saver.restore(sess, "saved_nn/save_net.ckpt")
        
        #print(testdata)
        #print(sess.run(tf.argmax(pred,1),feed_dict={X:testdata}))
        #print(sess.run(tf.argmax(pred,1),feed_dict={X:dataT,y:targetT}))
        result = sess.run(pred,feed_dict={X:testdata})
        category = 0
        if result[0][1] > result[0][0] and result[0][1] > result[0][2]:
            category = 1
        elif result[0][2] > result[0][0] and result[0][2] > result[0][1]:
                category = 2
    
        if (category == 0):
            labelMessage.config(text="'Excellent' grade predicted, keep up the good work!", font=("", 18), 
                                    fg='Green')

        elif (category == 1):
            labelMessage.config(text="     'Average' grade predicted, work harder!", font=("", 18), 
                                    fg='#F58D25')
        elif (category == 2):
                
            labelMessage.config(text="'Below Average' grade predicted, you must work harder!", font=("", 18), 
                                    fg='Red')
    

#-------------------------------------------------  ANFIS model   ------------------------------------
def button4Listener():
    #messagebox.showinfo("Title","ANNModel")
    A1 = float(text_A1.get())
    A2 = float(text_A2.get())
    A3 = float(text_A3.get())
    Q1 = float(text_Q1.get())
    Q2 = float(text_Q2.get())
    arr = (A1,A2,A3,Q1,Q2,0.0)
    #print(type(arr))

    eng = matlab.engine.start_matlab()
    ans = float(eng.ANFIS_predictmodel(arr))
    output = ans 
    '''print(type(ans))
    print(type(output))'''   

    if(output==3.0):
        labelMessage.config(text="Grade: Below Average, must work harder!",font=("",18),fg='red')
    elif(output==2.0):
        labelMessage.config(text="Grade: Average, work harder",font=("",18),fg='#F58D25')
    else:
        labelMessage.config(text=" Grade: Excellent , keep up the good work!",font=("",18),fg='Green')

#------------------------------------------ UI part ------------------------------------------------------
labelWelcome = Label(window, text="", bg='#D3ECF5')
label_A1 = Label(window, text = "Assignment#1:", bg='#D3ECF5',font=("",15))
label_A2 = Label(window,text = "Assignment#2:", bg='#D3ECF5',font=("",15))
label_A3 = Label(window,text = "Assignment#3:", bg='#D3ECF5',font=("",15))
label_Q1 = Label(window,text = "Quiz#1:", bg='#D3ECF5',font=("",15))
label_Q2 = Label(window,text = "Quiz#2:", bg='#D3ECF5',font=("",15))
label_empty = Label(window,text = "", bg='#D3ECF5',font=("",15))
                 
text_A0 = Label(window,text = "", bg='#D3ECF5',font=("",15))        
text_A1 = Entry(window,width=30)
text_A2 = Entry(window,width=30)
text_A3 = Entry(window,width=30)
text_Q1 = Entry(window,width=30)
text_Q2 = Entry(window,width=30)


labelWelcome.grid(row = 0,column = 0)
label_A1.grid(row = 1,column = 0)
label_A2.grid(row = 2,column = 0)
label_A3.grid(row = 3,column = 0)
label_Q1.grid(row = 4,column = 0)
label_Q2.grid(row = 5,column = 0)
label_empty.grid(row = 6,column = 0)

text_A0.grid(row = 0,column = 1)
text_A1.grid(row = 1,column = 1)
text_A2.grid(row = 2,column = 1)
text_A3.grid(row = 3,column = 1)
text_Q1.grid(row = 4,column = 1)
text_Q2.grid(row = 5,column = 1)
#text_Q2.grid(row = 5,column = 1)
        


#button_ok = Button(text = "Predict by NN",width = 30, height=2, command=button1Listener, bg='#62CCF0', font=("",15))
#button_anfis1 = Button(text = "Predict by ANFIS",width = 30, height=2, command=button2Listener, bg='#62CCF0', font=("",15))
button_ok2 = Button(text="Predict by NN Model",width=30,height=2,command=button3Listener,bg='#62CCF0', font=("",15))
button_anfis2 = Button(text = "Predict by ANFIS Model",width = 30, height=2, command=button4Listener, bg='#62CCF0', font=("",15))


#button_ok.place(x=80,y=120)       
#button_ok.grid(row = 7,column = 0)
#button_anfis1.grid(row = 7,column = 1)
button_ok2.grid(row = 7,column = 0)
button_anfis2.grid(row = 7,column = 1)

labelMessage = Label(window, text="Predict Your Final Grade now!", bg='#D3ECF5', font=("",25))
labelMessage.place(x=15, y=300)


window.mainloop()








