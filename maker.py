import numpy as np
import pickle
import matplotlib.pyplot as plt
hist=pickle.load(open("histlast"))
histnokori=range(500)
aki=np.zeros(500)
kuu=np.zeros(1000)
lastlist=np.zeros(500)
while 0 in lastlist :
    k=np.zeros(500)
    a=np.array([hist[i].max() for i in histnokori])
    print str(a.argmax()),"=",str(a.max()),"is max"
    point=hist[a.argmax()]
    print str(point.argmax()),"is selected"
    if point.argmax()==500:
        break
    if aki[point.argmax()]!=1:
        print str(point.argmax()),"is emp"
        aki[point.argmax()]=1
        lastlist[point.argmax()]=a.argmax()
        hist[a.argmax()]=kuu
    else:
        print str(point.argmax()),"is not emp"
        point[point.argmax()]=-1
e=open("filesA.txt")
k=open("nexA.txt","w")
for i in e:
    aa=i.split()
    ss=str(aa[0])+" "+str(int(lastlist[int(aa[1])]))+"\n"
    k.write(ss)
e.close()
k.close()
e=open("valA.txt")
k=open("nexvalA.txt","w")
for i in e:
    aa=i.split()
    ss=str(aa[0])+" "+str(int(lastlist[int(aa[1])]))+"\n"
    k.write(ss)
e.close()
k.close()

