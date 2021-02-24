from sklearn import metrics
import numpy
import csv
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import KFold
from train_model import DeepNet
import pandas
import matplotlib.pyplot as plt


"""
    Saves the embeddings to a file.
"""
def save_projections(encoded, base_name, ids):
    f_name = base_name + "_projection.csv"

    encoded = pandas.DataFrame(encoded)
    encoded.insert(0, "entrezId", ids)
    encoded["entrezId"] = pandas.to_numeric(encoded["entrezId"], downcast="integer")
    encoded = encoded.set_index("entrezId")

    print(f_name)
    encoded.to_csv(f_name, index=True)


class CrossValidation:

    def __init__(self, dataset, classifier_builder, classifier_params, base_folder):
        self.dataset = dataset
        self.classifier_builder = classifier_builder
        self.classifier_params = classifier_params
        self.base_folder = base_folder

    @staticmethod
    def copy_properties(ds, train, test):
        train.binary_features = ds.binary_features
        train.class_indexes = ds.class_indexes
        train.feature_indexes = ds.feature_indexes

        test.binary_features = ds.binary_features
        test.class_indexes = ds.class_indexes
        test.feature_indexes = ds.feature_indexes

    @staticmethod
    def write_results(ids, predictions, classes, fold_id, fold_file_path):



        auroc = metrics.roc_auc_score(classes, predictions, average="micro")
        # h = metrics.hamming_loss(classes, predictions)#汉明损失
        c = metrics.coverage_error(classes, predictions)  # - 1  # 减 1原因：看第2个参考链接覆盖损失
        r = metrics.label_ranking_loss(classes, predictions)#排名损失
        a = metrics.average_precision_score(classes, predictions)#平均精度损失
        #if auroc>0.91 and auroc<0.925:
        if fold_id>-1:

            dataframe = pandas.DataFrame(classes)
            dataframe.to_csv('fulldataset0_classes.csv',sep=',',index=False,mode='a+',header=False)
            dataframe1 = pandas.DataFrame(predictions)
            dataframe1.to_csv('fulldataset0_predictions.csv', sep=',', index=False, mode='a+',header=False)
        # numpy.savetxt('classes'+str(fold_id)+'.csv', classes, delimiter=',')
        # numpy.savetxt('predictions'+str(fold_id)+'.csv', predictions, delimiter=',')
        #
        # with open("fulldataset.csv", "a") as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerows(classes)

        # accur=metrics.accuracy_score(classes,predictions)#####################################################################

        # fpr = dict()
        # print("###############################################"+str(classes.shape[1])+str(predictions.shape[1]))
        # tpr = dict()
        # roc_auc = dict()
        # for i in range(27):
        #     fpr[i], tpr[i], _= roc_curve(classes.values[:, i], predictions.values[:, i])
        #     roc_auc[i] = auc(fpr[i], tpr[i])
        # # Compute micro-average ROC curve and ROC area
        # fpr["micro"], tpr["micro"], _= roc_curve(classes.values.ravel(), predictions.values.ravel())
        # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # plt.figure()
        # plt.plot(fpr["micro"], tpr["micro"],
        #          label='micro-average ROC curve (area = {0:0.2f})'
        #                ''.format(roc_auc["micro"]),
        #          color='deeppink', linestyle=':', linewidth=4)
        # plt.plot([0, 1], [0, 1], 'k--', lw=2)
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Some extension of Receiver operating characteristic to multi-class')
        # plt.legend(loc="lower right")
        # plt.show()



        with open(fold_file_path + "predictions_fold_" + str(fold_id)+ ".csv" , "w") as f:
            f.write("AUROC," + str(auroc)+',' + str(fold_id)+"\n")
            f.write("覆盖损失,"+str(c)+','+"排名损失,"+str(r)+','+"平均精度损失,"+str(a)+'\n')#"汉明损失,"+str(h)+','+
            f.write("geneID"+ ","+"class_Brain.Alzheimer,"+" ,"+"class_Brain.Multiple.Sclerosis,"+" ,"+"class_Brain.Parkinson,"+" ,"+"class_Disease.Brain,"+" ,"+"class_Disease.Heart,"+" ,"+"class_Disease.Immune,"+" ,"
                    +"class_Disease.Muscle,"+" ,"+"class_Disease.Neoplasm,"+" ,"+"class_Disease.Nutrition,"+" ,"+"class_Heart.Arteriosclerosis,"+" ,"+"class_Heart.Coronary.Disease,"+" ,"+"class_Heart.Hypertension,"+" ,"
                    +"class_Heart.Myocardial.Infarction,"+" ,"+"class_Immune.Hypersensitivity,"+" ,"+"class_Muscle.Arthritis,"+" ,"+"class_Muscle.Osteoporosis,"+" ,"+"class_Neoplasm.Adenocarcinoma,"+" ,"+"class_Neoplasm.Breast,"+" ,"
                    +"class_Neoplasm.Colorectal,"+" ,"+"class_Neoplasm.Lung,"+" ,"+"class_Neoplasm.Prostatic,"+" ,"+"class_Neoplasm.Stomach,"+" ,"+"class_Nutritional.Diabetes.Type1,"+" ,"+"class_Nutritional.Diabetes.Type2,"+" ,"
                    +"class_Nutritional.Obesity,"+" ,"+"class_Respiratory.Asthma,"+" ,"+"class_Disease,"+"\n")

            # f.write("accur, "+ str(accur)+"\n")##################################################

            for i in range(predictions.shape[0]):  # for each instance
                id_instance = ids[i]
                f.write(str(id_instance) + ",")

                pred = predictions.iloc[i] # predicted class probability
                clas = classes.iloc[i] # actual class

                for j in range(clas.shape[0]):  # for each class
                    # class_name = clas.index[j]
                    f.write(str(pred[j]) + "," + str(clas[j]))
                    if j != clas.shape[0] - 1:
                        f.write(",")

                f.write("\n")


    """
        Writes the indexes of the instances used in each fold in a file.
    """
    @staticmethod
    def write_folds(train, test, f_name):
        with open(f_name, "w") as f:
            f.write(str(list(train.index.tolist())) + "\n")
            f.write(str(list(test.index.tolist())) + "\n")

    """
        Does the actual cross-validation.
        If fold_to_run is provided, will run the 10-cv for that fold only.
    """
    def run(self, fold_to_run=None):

        kf = KFold(n_splits=10, shuffle=True)##########################################



        cur_fold = 0
        # for each fold
        for train_index, test_index in kf.split(self.dataset):

            if fold_to_run is not None and fold_to_run != cur_fold:
                cur_fold += 1
                continue

            # gets the training and testing sets
            train = self.dataset.iloc[train_index, :]
            test = self.dataset.iloc[test_index, :]

            # copies the speciall properties in self.dataset to train and test. Writes the ids of the instances in a file.
            CrossValidation.copy_properties(self.dataset, train, test)
            CrossValidation.write_folds(train, test, self.base_folder + "/folds/fold_" + str(cur_fold))

            # induces the classifier
            cla = self.classifier_builder(**self.classifier_params)
            cla.train(train, test, lc_file=self.base_folder + "/lcs/lc_fold_" + str(cur_fold))

            # gets the predictions (for the testing set and the training set)
            ids_test, predictions_test, classes_test = cla.evaluate(test)
            ids_train, predictions_train, classes_train = cla.evaluate(train)

            # writes the auroc and predictions in a file.
            CrossValidation.write_results(ids_train, predictions_train, classes_train, cur_fold, self.base_folder + "/predictions_train/")
            CrossValidation.write_results(ids_test, predictions_test, classes_test, cur_fold, self.base_folder + "/predictions/")

            # do extra stuff if the classifier is a DNN
            if isinstance(cla, DeepNet):
                # gets and saves the global embedings (regardless of the classes)
                ids, projection, classes = cla.get_projection(train)
                save_projections(projection, self.base_folder + "/projections/fold_train_" + str(cur_fold), ids)

                # gets the per-class projections (by multiplying the output with the weights in the last layer)
                for class_name in  classes_train.columns:
                    ids, projection, classes = cla.get_projection(train, class_name)
                    save_projections(projection, self.base_folder + "/projections/class_fold_train_" + class_name + "_" + str(cur_fold), ids)


                ids, projection, classes = cla.get_projection(test)
                save_projections(projection, self.base_folder + "/projections/fold_test_" + str(cur_fold), ids)

                for class_name in  classes_train.columns:
                    ids, projection, classes = cla.get_projection(test, class_name)
                    save_projections(projection, self.base_folder + "/projections/class_fold_test_" + class_name + "_" + str(cur_fold), ids)

            # saves the classification model
            cla.save_model(self.base_folder + "/models/model_" + str(cur_fold))

            cur_fold += 1


