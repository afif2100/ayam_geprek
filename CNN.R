# Convolutional Neural Networks

# Load packages
library(keras)
library(EBImage)



##### Read data train----------------------------------------------------------------
setwd("D://Workshop/keras on R/")

dir_path <- "data/train/"
img_name <- list.files(dir_path)
length(img_name)

#read_image
train <- list()
for (i in 1:length(img_name)) {
  train[[i]] <- readImage(paste(dir_path,img_name[i],sep=""))
  }

#rezise
for (i in 1:length(img_name)) {
  train[[i]] <- resize(train[[i]], 100, 100)
  }

##### Read data test----------------------------------------------------------------
dir_path <- "data/test/"
img_name <- list.files(dir_path)
length(img_name)

#read_image
test <- list()
for (i in 1:length(img_name)) {
  test[[i]] <- readImage(paste(dir_path,img_name[i],sep=""))
}

#rezise
for (i in 1:length(img_name)) {
  test[[i]] <- resize(test[[i]], 100, 100)
}


##### Save Data----------------------------------------------------------------
{
saveRDS(train,file = "train.RDS")
saveRDS(test,file = "test.RDS")
}

##### Load Data----------------------------------------------------------------
{
  train <- readRDS(file = "train.RDS")
  test <- readRDS(file = "test.RDS")
}


#####-----------------------------------------------------------------


#combine
trainc <- combine(train)
testc <- combine(test)
dim(trainc);dim(testc)

# Reorder dimension
train <- aperm(trainc, c(4, 1, 2, 3))
test <- aperm(testc, c(4, 1, 2, 3))
dim(train);dim(test)


#label
labeling <- function(kelas,data){
label <-c()
n = dim(data)[1]/kelas
for (i in 1:kelas) {
   label <- c(label,c(rep(i-1,n)))
}
return(label)
}

trainy <- labeling(kelas= 3,data= train)
testy <- labeling(kelas= 3,data= test)


#to data categorical
trainLabels <- to_categorical(trainy)
testLabels <- to_categorical(testy)
tail(trainLabels);tail(testLabels)



#####-------------------------------------

# Model
{
model <- keras_model_sequential()

model %>%
  #definisi layer konvolusi
  layer_conv_2d(filters = 84 , 
                kernel_size = c(5,5),
                activation = 'relu',
                input_shape = c(100,100,3)) %>%
  layer_conv_2d(filters = 64,
                kernel_size = c(5,5),
                activation = 'relu') %>%
  layer_average_pooling_2d(pool_size = c(5,5)) %>%
  

  layer_conv_2d(filters = 32,
                kernel_size = c(3,3),
                activation = 'relu') %>%
  layer_conv_2d(filters = 32,
                kernel_size = c(3,3),
                activation = 'relu') %>%
  layer_conv_2d(filters = 32,
                kernel_size = c(3,3),
                activation = 'relu') %>%
  layer_conv_2d(filters = 32,
                kernel_size = c(3,3),
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(3,3)) %>%
  layer_dropout(rate = 0.5) %>%
  
  
  
  ## flaten layer jadi neural network biasa
  layer_flatten() %>%
  layer_dense(units = 512, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  
  layer_dense(units = 256, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  
  layer_dense(units = 256, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%

  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  

  
  layer_dense(units = 64, activation = 'tanh') %>%

    ## layer output ada 3 kategori
  layer_dense(units = 3, activation = 'softmax')


### compile model ---------------------------------
  
## mendefinisikan optimizer
opt <-optimizer_adam(lr=0.0001, 
                     beta_1=0.9, 
                     beta_2=0.999, 
                     decay=0)
## compile model
model %>% compile(loss = 'categorical_crossentropy',
          optimizer = opt,
          metrics = c('accuracy'))

summary(model)

##### Training-------------------------------------

  
# Fit model
history <- model %>%
  fit(train,
      trainLabels,
      epochs = 300,
      batch_size =30,
      shuffle = TRUE,
      validation_split = 0.2,
      validation_data = list(test, testLabels))

plot(history)

}


tensorboard(log_dir = "")

# Evaluation & Prediction - train data
model %>% evaluate(train, trainLabels)
pred <- model %>% predict_classes(train)
table(Predicted = pred, Actual = trainy)

prob <- model %>% predict_proba(train)
cbind(prob, Predicted_class = pred, Actual = trainy)

# Evaluation & Prediction - test data
model %>% evaluate(test, testLabels)
pred <- model %>% predict_classes(test)
table(Predicted = pred, Actual = testy)

prob <- model %>% predict_proba(test)
cbind(prob, Predicted_class = pred, Actual = testy)

##### save model---------------------------------------------

#save model
save_model_weights_hdf5(model,filepath='test.hdf5',overwrite=TRUE)
#load model
model=load_model_weights_hdf5(model,filepath="test.hdf5",by_name=FALSE)