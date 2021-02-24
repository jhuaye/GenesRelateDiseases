import os
import re
c=0
# path="E:\home\fabiofabris\results_proj_2\go+pathdip+ppi+gtex\predictions"
# savepath="E:\home\fabiofabris\results_proj_2\go+pathdip+ppi+gtex\predictions"

# filelist = os.listdir(path)
f1 = open("./predictions_fold_-1.csv")
line1 = f1.readline()
res=open("./res.txt","a")
res.write(line1)
for a in range(0,10):

    f = open("./predictions_fold_"+str(a)+".csv")
    line = f.readline()
    b=re.sub("AUROC,", "", line)
    b = re.sub(str(a)+"\n", "", b)
    b = float(re.sub(",", "", b))
    c=c+b
    res=open("./res.txt","a")
    res.write(line)

res=open("./res.txt","a")
res.write(str(c/10)+'\n')

