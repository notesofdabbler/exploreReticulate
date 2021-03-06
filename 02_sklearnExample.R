#
# Example of using scikit learn from R using reticulate
#

# load library
library(reticulate)
library(plotly)

use_python("/Users/shanki/anaconda3/bin/python", required = TRUE)
py_config()

main = import_main()
bi = import_builtins()
np = import("numpy")
sklearn = import("sklearn")

#   MNIST digit classification with Naive Bayes
#  This portion is from the book:
#  Python Datascience Handbook - Jake Vanderplas 
#  https://jakevdp.github.io/PythonDataScienceHandbook/05.02-introducing-scikit-learn.html
#
digits = sklearn$datasets$load_digits()
data = py_to_r(py_get_attr(digits, "data"))
dim(data)
target = py_to_r(py_get_attr(digits, "target"))
dim(target)

train = sample(1:length(target), round(0.8*length(target)))
length(train)

data_train = data[train,]
target_train = target[train]

data_test = data[-train,]
target_test = target[-train]

GaussianNB = import("sklearn.naive_bayes")$GaussianNB

model = GaussianNB()
model$fit(data_train, target_train)
y_model = model$predict(data_test)

accuracy_score = import("sklearn.metrics")$accuracy_score

acc = accuracy_score(y_model, target_test)
acc

confusion_matrix = import("sklearn.metrics")$confusion_matrix

confmat = confusion_matrix(y_model, target_test)

plot_ly(x = as.character(seq(1, 10)), y = as.character(seq(1, 10)),
         z = confmat, type = "heatmap")


# Use the seaborn plot code as is
py_run_string("

import matplotlib.pyplot as plt
import seaborn as sns

def pltconf(confmat):
    sns.heatmap(confmat, square=True, annot=True, cbar=False)
    plt.xlabel('predicted value')
    plt.ylabel('true value')
    plt.show()
                        ")

py$pltconf(confmat)

# MNIST digit classification with Random Forest
# https://jakevdp.github.io/PythonDataScienceHandbook/05.08-random-forests.html
#

RandomForestClassifier = import("sklearn.ensemble")$RandomForestClassifier

model = RandomForestClassifier(n_estimators = 1000L)
model$fit(data_train, target_train)
ypred = model$predict(data_test)

cat(sklearn$metrics$classification_report(ypred, target_test))

confmat = confusion_matrix(y_model, target_test)

plot_ly(x = as.character(seq(1, 10)), y = as.character(seq(1, 10)),
        z = confmat, type = "heatmap")

main$pltconf(confmat)
