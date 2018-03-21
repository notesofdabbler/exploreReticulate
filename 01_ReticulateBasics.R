# based of
# https://rstudio.github.io/reticulate/articles/calling_python.html

# load library
library(reticulate)

# specifiy python configuration to use
# For some reason if I try running this after py_config(), it doesn't take effect
use_python("/Users/shanki/anaconda3/bin/python", required = TRUE)
# check python configuration
py_config()

# import main and builtin modules
main = import_main()
bi = import_builtins()

# import numpy
np = import("numpy")

# example of setting/getting directories using python os module
os = import("os")
os$getcwd()
os$chdir("/Users")
getwd()

# get help on a python function
py_help(os$chdir)

getwd()
setwd("~/Documents/notesofdabbler/exploreReticulate/")
os$getcwd()

# work with python functions, example is string splitting
x = "a b c"

# r_to_py converts x to a python string
bi$type(r_to_py(x))
r_to_py(x)$split() # not quite a typical R object yet
class(r_to_py(x)$split())

py_to_r(r_to_py(x)$split()) # manual conversion to R character object
class(py_to_r(r_to_py(x)$split()))

# run python code using py_run_string
py_run_string("x = 10")
py$x

py_run_string("y = x * 2")
py$y

# example using numpy
np$arange(10)
x = np$cumsum(np$arange(10))
x
class(x)

py_help(np$arange)

x = matrix(seq(1,12), nrow = 4)
x
np_array(x)
class(np_array(x))

x = np$reshape(seq(1,12), c(3,4))
x

x = np$reshape(seq(1,12), c(3L,4L), order = "C")
x

# iterators
iterate(bi$iter(c(1,2,3,4)), print)
iterate(bi$iter('abcd'), print)

x = bi$iter(c(1,2,3,4))
while (TRUE) {
  item = iter_next(x)
  print(item)
  if(is.null(item)) {
    break
  }
}

x = bi$iter(bi$range(1000L))
while (TRUE)  {
  val = iter_next(x)
  print(val)
  if (val > 100) {
    break
  }
}

# define a generator function
sequence_generator <-function(start) {
  value <- start
  function() {
    value <<- value + 1
    value
  }
}

# convert the function to a python iterator
iter <- py_iterator(sequence_generator(10))

iter_next(iter)
iter_next(iter)
iter_next(iter)

# movement between r dataframe and pandas dataframe
data(mtcars)
head(mtcars)
summary(mtcars)

pd = import("pandas")
bi$type(r_to_py(mtcars))

pymtcars = r_to_py(mtcars)
pymtcars$head()
pymtcars$describe()

cntrypop = pd$read_html("http://www.worldometers.info/world-population/population-by-country/")
cntrypopdf = py_to_r(cntrypop[[1]])

library(ggplot2)
library(dplyr)
ggplot(cntrypopdf %>% arrange(desc(`Population (2018)`)) %>% slice(1:20)) + 
  geom_bar(aes(x = reorder(`Country (or dependency)`, `Population (2018)`), y = `Population (2018)`), stat = "identity") + 
   coord_flip() + xlab("") + theme_bw()