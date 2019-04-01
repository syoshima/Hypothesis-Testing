  install.packages('plyr') # we need this package for computing frequency counts
  
  library(plyr)
  # Clear variables.
  rm(list = ls())  
  # Set Directory
  setwd("C:/Users/Samy Abud Yoshima/Anaconda3/Library/Courses/MIT XPRO/DataScience+BigData/Module 3 - HypTest and Classification/CaseSt 3.1")
  
  
  data <- read.csv("challenger-data.csv")
  
  
  failures <- subset(data, data$Y == 1) # subset of the data for just failures
  no_failures <- subset(data, data$Y == 0) # subset of the data for no failures
  failures_freq <- count(failures, 'X') # count of failures, for each temperature
  no_failures_freq <- count(no_failures, 'X') # count of no failures, for each temperature
  plot(no_failures$X, integer(length(no_failures$X)), ylim=c(-0.5, 5), col='blue', xlab='X:Temperature', ylab='Number of Failures', pch=19) # plot the no failures first
  points(failures_freq$X, failures_freq$freq, col='red', pch=19) #add the failures
  model = glm(data$Y ~ data$X,family=binomial(link='logit'),data=data) # build the model
  summary(model)