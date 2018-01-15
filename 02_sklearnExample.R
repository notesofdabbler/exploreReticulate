
library(reticulate)

use_python("/Users/shanki/anaconda3/bin/python", required = TRUE)
py_config()

main = import_main()
py = import_builtins()
np = import("numpy")
sklearn = import("sklearn")

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

model = sklearn$naive_bayes$GaussianNB()
model$fit(data_train, target_train)
y_model = model$predict(data_test)

acc = sklearn$metrics$accuracy_score(y_model, target_test)
acc

model = sklearn$ensemble$RandomForestClassifier(n_estimators = 1000L)
model$fit(data_train, target_train)
ypred = model$predict(data_test)

cat(sklearn$metrics$classification_report(ypred, target_test))
