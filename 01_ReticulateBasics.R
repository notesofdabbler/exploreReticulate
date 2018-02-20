
# load library
library(reticulate)

# check python configuration
py_config()
# specifiy python configuration to use
use_python("/Users/shanki/anaconda3/bin/python", required = TRUE)

# import main and builtin modules
main = import_main()
py = import_builtins()

# import numpy
np = import("numpy")

# example of setting/getting directories using python os module
os = import("os")
os$getcwd()
os$chdir("/Users")

py_help(os$chdir)

getwd()
setwd("~/Documents/notesofdabbler/exploreReticulate/")
os$getcwd()

# work with python functions, example is string splitting
x = "a b c"
r_to_py(x)$split()
class(r_to_py(x)$split())

py_to_r(r_to_py(x)$split())
class(py_to_r(r_to_py(x)$split()))

# run python code using py_run_string
main = py_run_string("x = 10")
main$x

main = py_run_string("y = x * 2")
main$y

# example using numpy
np$arange(10)
x = np$cumsum(np$arange(10))
class(x)

py_help(np$arange)

x = matrix(seq(1,12), nrow = 4)
x
np_array(x)
class(np_array(x))

x = np_array(seq(1,12), c(3,4))
x

x = np_array(seq(1,12), c(3,4), order = "C")
x

# iterators
iterate(py$iter(c(1,2,3,4)), print)
iterate(py$iter('abcd'), print)

x = py$iter(c(1,2,3,4))
while (TRUE) {
  item = iter_next(x)
  print(item)
  if(is.null(item)) {
    break
  }
}

x = py$iter(py$range(1000L))
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

