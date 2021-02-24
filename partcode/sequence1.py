import pandas as pd
import csv
list=['class_Brain.Alzheimer','class_Brain.Multiple.Sclerosis','class_Brain.Parkinson','class_Disease.Brain', 'class_Disease.Heart',
      'class_Disease.Immune' ,'class_Disease.Muscle','class_Disease.Neoplasm','class_Disease.Nutrition','class_Heart.Arteriosclerosis' ,
      'class_Heart.Coronary.Disease','class_Heart.Hypertension','class_Heart.Myocardial.Infarction','class_Immune.Hypersensitivity' ,
      'class_Muscle.Arthritis', 'class_Muscle.Osteoporosis', 'class_Neoplasm.Adenocarcinoma', 'class_Neoplasm.Breast', 'class_Neoplasm.Colorectal',
      'class_Neoplasm.Lung' ,'class_Neoplasm.Prostatic', 'class_Neoplasm.Stomach', 'class_Nutritional.Diabetes.Type1', 'class_Nutritional.Diabetes.Type2'
         ,'class_Nutritional.Obesity', 'class_Respiratory.Asthma' ,'class_Disease'
]


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False
for i in list:
    b = pd.read_csv('test1/' + i + ".csv", sep=',')
    d=b.sort_values(by='genes',axis=0,inplace=False,ascending=False)
    # c=b.tolist()
    a=''
    c=0
    e=[0]
    f=1
    with open("result1/"+i+".csv", "a+",newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["genesid", "average","max","min","times"])
        for i, j in d.iterrows():
            if a!=j[0] and is_number(j[1]):
                print('gene'+str(a)+':')
                writer.writerow([a,c/f,max(e),min(e),f])
                a=j[0]
                print("min: "+str(min(e))+"max: "+str(max(e))+"avg: "+str(c/f)+"\n")
                e = []
                f=1
                e.append(float(j[1]))
                c=float(j[1])
            elif  is_number(j[1]):
                f=f+1
                e.append(float(j[1]))
                c=c+float(j[1])

for i in list:
    b = pd.read_csv('result1/' + i + ".csv", sep=',')
    d=b.sort_values(by='average',axis=0,inplace=False,ascending=False)
    with open("results1/" + i + ".csv", "a+", newline="") as csvfile:
        d.to_csv("results1/" + i + ".csv")