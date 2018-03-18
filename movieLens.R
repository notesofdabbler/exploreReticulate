#
# Based of Jeremy Howard's deep learning course part 1 jupyter notebook
#  https://github.com/fastai/fastai/blob/master/courses/dl1/lesson5-movielens.ipynb
#

# import R libraries
library(reticulate)
library(ggplot2)
library(dplyr)

use_python("/home/paperspace/anaconda3/envs/fastai/bin/python", required = TRUE)
use_condaenv("fastai")
py_config()

main = import_main()
bi = import_builtins()

# get relevant python imports
fstai_learner = import_from_path("fastai.learner", "../../fastai")
fstai_coldata = import_from_path("fastai.column_data", "../../fastai")

py_run_string("
from fastai.learner import *
from fastai.column_data import *
              ")

datapath = "../../data/ml-latest-small/"

ratings = read.csv(paste0(datapath, "ratings.csv"), stringsAsFactors = FALSE)
head(ratings)

movies = read.csv(paste0(datapath, "movies.csv"), stringsAsFactors = FALSE)
head(movies)

val_idxs = py$get_cv_idxs(nrow(ratings))
val_idxs = as.integer(val_idxs)
wd = 2e-4
n_factors = 50L

cf = py$CollabFilterDataset$from_csv(datapath, 'ratings.csv', 'userId', 'movieId', 'rating')

learn = cf$get_learner(n_factors, val_idxs, 64L, opt_fn=py$optim$Adam)

learn$fit(1e-2, 2L, wds=wd, cycle_len=1L, cycle_mult=2L)

preds = learn$predict()

y=learn$data$val_y

dfplt = data.frame(y = y, preds = preds)
ggplot(dfplt) + geom_hex(aes(x = preds, y = y), alpha = 0.5)

m = learn$model
m$cuda()

movie2idx = cf$item2idx
movie2idx = unlist(movie2idx)
topmovies = ratings %>% group_by(movieId) %>% summarize(cnt = n()) %>% arrange(desc(cnt)) %>% slice(1:3000)
topmoviesidx = movie2idx[as.character(topmovies$movieId)]
topmoviesidx = np_array(topmoviesidx)

movie_bias = py$to_np(m$ib(py$V(topmoviesidx)))
movie_bias[1:20]

topmovies$movie_bias = movie_bias[,1]
topmovies = left_join(topmovies, movies %>% select(movieId, title), by = "movieId")

topmovies %>% arrange(movie_bias) %>% slice(1:15)
topmovies %>% arrange(desc(movie_bias)) %>% slice(1:15)

Amat = matrix(c(1.0, 2.0, 3.0, 4.0), byrow = TRUE, ncol = 2)
Bmat = matrix(c(2.0, 2.0, 10.0, 10.0), byrow = TRUE, ncol = 2)

a = py$T(Amat)
a
b = main$T(Bmat)
b

a$mul(b)

a$mul(b)$sum(1L) # note the use of 1L instead of 1 to ensure that we are passing integer

py_run_string("
class DotProduct(nn.Module):
    def forward(self, u, m): return (u*m).sum(1)              
              ")

model = main$DotProduct()
model$forward(a, b)

u_unique = unique(ratings$userId)
user2idx = as.integer(seq(0, length(u_unique) - 1))
names(user2idx) = u_unique
user2idx[1:10]

m_unique = unique(ratings$movieId)
movie2idx = as.integer(seq(0, length(m_unique) - 1))
names(movie2idx) = m_unique
movie2idx[1:10]

ratings$userId = user2idx[as.character(ratings$userId)]
ratings$movieId = movie2idx[as.character(ratings$movieId)]

n_users = length(u_unique)
n_movies = length(m_unique)

py_run_string("
class EmbeddingDot(nn.Module):
    def __init__(self, n_users, n_movies, n_factors):
        super().__init__()
        self.u = nn.Embedding(n_users, n_factors)
        self.m = nn.Embedding(n_movies, n_factors)
        self.u.weight.data.uniform_(0,0.05)
        self.m.weight.data.uniform_(0,0.05)
        
    def forward(self, cats, conts):
        users,movies = cats[:,0],cats[:,1]
        u,m = self.u(users),self.m(movies)
        return (u*m).sum(1)              
              ")

py_run_string("
def getdata(datapath, userId, movieId, rating):
    val_idxs = get_cv_idxs(len(rating))
    ratings = pd.DataFrame({'userId':userId, 'movieId':movieId, 'rating':rating})
    x = ratings.drop(['rating'],axis=1)
    y = ratings['rating'].astype(np.float32)
    data = ColumnarModelData.from_data_frame(datapath, val_idxs, x, y, ['userId', 'movieId'], 64)
    return data
              ")

data = py$getdata(datapath, as.integer(ratings$userId), as.integer(ratings$movieId), ratings$rating)

wd = 1.0e-5
n_factors = 50L
model = py$EmbeddingDot(as.integer(n_users), as.integer(n_movies), n_factors)$cuda()
opt = py$optim$SGD(model$parameters(), 1e-1, weight_decay=wd, momentum=0.9)
py$fit(model, data, 3L, opt, py$F$mse_loss)

py$set_lrs(opt, 0.01)

py$fit(model, data, 3L, opt, py$F$mse_loss)

py_run_string("
def get_emb(ni,nf):
    e = nn.Embedding(ni, nf)
    e.weight.data.uniform_(-0.01,0.01)
    return e
              
class EmbeddingDotBias(nn.Module):
    def __init__(self, n_users, n_movies):
        super().__init__()
        (self.u, self.m, self.ub, self.mb) = [get_emb(*o) for o in [
              (n_users, n_factors), (n_movies, n_factors), (n_users,1), (n_movies,1)
              ]]
              
    def forward(self, cats, conts):
        users,movies = cats[:,0],cats[:,1]
        um = (self.u(users)* self.m(movies)).sum(1)
        res = um + self.ub(users).squeeze() + self.mb(movies).squeeze()
        res = F.sigmoid(res) * (max_rating-min_rating) + min_rating
        return res              
              ")

wd=2e-4
model = py$EmbeddingDotBias(cf.n_users, cf.n_items).cuda()
opt = optim.SGD(model.parameters(), 1e-1, weight_decay=wd, momentum=0.9)
