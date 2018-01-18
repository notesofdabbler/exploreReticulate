#
# https://github.com/fastai/fastai/blob/master/tutorials/linalg_pytorch.ipynb
#

library(reticulate)

fastai = import("fastai")
dsets = import("fastai.dataset")
pickle = import("pickle")
gzip = import("gzip")

main = import_main()
py = import_builtins()

FILENAME = "data/mnist.pkl.gz"
filename = paste0(URL, FILENAME)


mnist_data = pickle$load(gzip$open(FILENAME, 'rb'), encoding='latin-1')

x = mnist_data[[1]][[1]]
y = mnist_data[[1]][[2]]

x_valid = mnist_data[[2]][[1]]
y_valid = mnist_data[[2]][[2]]

dim(x); dim(y); dim(x_valid); dim(y_valid)

xmean = mean(x)
xstd = sd(x)

x = (x - xmean)/xstd
mean(x); sd(x)

x_valid = (x_valid - xmean)/xstd

path = "data/"

md = fastai$dataset$ImageClassifierData$from_arrays(path, tuple(x, y), tuple(x_valid, y_valid))


