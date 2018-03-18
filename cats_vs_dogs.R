
library(reticulate)
library(imager)

py_config()
use_python("/home/paperspace/anaconda3/envs/fastai/bin/python", required = TRUE)
use_condaenv("fastai")

main = import_main()
py = import_builtins()

fstai_imports = import_from_path("fastai.imports", "../../fastai/")
fstai_tfms = import_from_path("fastai.transforms", "../../fastai/")
fstai_convlearner = import_from_path("fastai.conv_learner", "../../fastai/")
fstai_model = import_from_path("fastai.model", "../../fastai/")
fstai_dataset = import_from_path("fastai.dataset", "../../fastai/")
fstai_sgdr = import_from_path("fastai.sgdr", "../../fastai/fastai")
fstai_plots = import_from_path("fastai.plots", "../../fastai/fastai")

torch = import("torch")
resnet34 = import("torchvision.models")$resnet34

PATH = "../../data/dogscats/"
sz = 224L

torch$cuda$is_available()
torch$backends$cudnn$enabled

list.files(PATH)
list.files(file.path(PATH,"valid"))
list.files(file.path(PATH, "valid/cats"))[1:5]

main = py_run_string("
from fastai.transforms import *
def mytfms(sz):
  return tfms_from_model(resnet34, sz)
")

data = fstai_dataset$ImageClassifierData$from_paths(PATH, tfms=py_call(fstai_tfms$tfms_from_model, resnet34, sz))
learn = fstai_convlearner$ConvLearner$pretrained(resnet34, data, precompute=TRUE)
learn$fit(0.01, 2L)
