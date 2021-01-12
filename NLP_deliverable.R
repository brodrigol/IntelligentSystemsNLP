library(tm)
library(textstem)
library(caret)

#LOADING THE DATA SETS
tweet=read.csv(url('https://raw.githubusercontent.com/brodrigol/IntelligentSystemsNLP/main/train.csv'), header = F, sep = ";")
test=read.csv(url('https://raw.githubusercontent.com/brodrigol/IntelligentSystemsNLP/main/test.csv'), header = F, sep = ";")
general=rbind(tweet,test)
names(general)=c("Tweets","Sentiment")

#Lemmatizing the test
for (i in 1:nrow(general)){
  general[i,1]=gsub(" t ", "t ", general[i,1])
  general[i,1]=gsub(" s ", " is ", general[i,1])
  general[i,1]=gsub(" re ", " are ", general[i,1])
  general[i,1]=lemmatize_strings(general[i,1])
}

#Creation of train and test set
train=sample(18000,0.8*18000)
tweet <- general[train,]
test <- general[-train,]
names(test)=c("Tweets","Sentiment")
names(tweet)=c("Tweets","Sentiment")

#Creation of dataframes for each emotion
tweet_anger=tweet[which(tweet$Sentiment=="anger"),]
tweet_fear=tweet[which(tweet$Sentiment=="fear"),]
tweet_joy=tweet[which(tweet$Sentiment=="joy"),]
tweet_love=tweet[which(tweet$Sentiment=="love"),]
tweet_sadness=tweet[which(tweet$Sentiment=="sadness"),]
tweet_surprise=tweet[which(tweet$Sentiment=="surprise"),]


#Preparing to create the corpus and preprocess data.
emotions=list(tweet_anger,tweet_fear,tweet_joy,tweet_love,tweet_sadness,tweet_surprise)
words=list()
double=list()
myStopwords=gsub("'","",stopwords()[-167])

#Create a corpus with the training data
j=1
for (i in emotions){
  corpus = iconv(i$`Tweets`, to = "UTF-8")
  corpus = Corpus(VectorSource(corpus))
  corpus <- tm_map(corpus, tolower)
  corpus=tm_map(corpus,removePunctuation)
  corpus=tm_map(corpus,removeNumbers)
  corpus=tm_map(corpus,removeWords,myStopwords)
  clean=tm_map(corpus,stripWhitespace)
  tdm=TermDocumentMatrix(clean)
  tdm=as.matrix(tdm)
  freq=rowSums(as.matrix(tdm))
  words[[j]]=sort(freq)/sum(freq)
  j=j+1
}
names(words)=c("Anger","Fear","Joy","Love","Sadness","Surprise")

#Remove words with high frequency for all emotions
list=c()
k=1
for (i in words) {
  for (j in 1:length(i)) {
    if (i[j]>=quantile(i)[4]){
      list[k]=names(i[j])
      k=k+1
    }
  }
}
stop=c()
for (i in list) {
  if (length(which(list==i))==6){
    stop=append(stop,i)
  }
}
stop=head(stop,n=length(stop)/6)
r=c()
for (i in 1:6){
  k=1
  r=c()
  for (j in stop){
    if (length(intersect(j,names(words[[i]])))>0) {
      r[k]=which(names(words[[i]])==j)
      k=k+1
    }
  }
  words[[i]]=words[[i]][-r]
}

#Provide a higher score to words specific for one emotion.
exclusive=list()
exclusive[[2]]=setdiff(names(words[[2]]),c(names(words[[3]]),names(words[[3]])))
for (j in 1:6) {
  other=which(c(1:6)!=j)
  exclusive[[j]]=setdiff(names(words[[j]]),c(names(words[[other[1]]]),names(words[[other[2]]]),names(words[[other[3]]]),names(words[[other[4]]]),names(words[[other[5]]])))
  int=intersect(exclusive[[j]],names(words[[j]]))
  words[[j]][int]=words[[j]][int]*7
}
names(exclusive)=c("Anger","Fear","Joy","Love","Sadness","Surprise")

#Improvements
#Provide a higher value to words specific for LOVE and a lower value to those that aren't specific.
love=setdiff(names(words$Love),c(names(words$Joy),names(words$Sadness)))
int=intersect(love,names(words$Love))
words$Love[int]=words$Love[int]*2
out=c(setdiff(love,names(words$Love)),setdiff(names(words$Love),love))
words$Love[out]=words$Love[out]/2

#Provide a higher value to words that differenciate Joy from Love
joy=setdiff(names(words$Joy),names(words$Love))
int=intersect(joy,names(words$Joy))
words$Joy[int]=words$Joy[int]*4

#Provide a higher value to words that differenciate Sadness from Love
sad=setdiff(names(words$Sadness),names(words$Love))
int=intersect(sad,names(words$Sadness))
words$Sadness[int]=words$Sadness[int]*4


#Prediction function.
prediction <- function(tweet_test) {
  tweet_words=c(strsplit(tweet_test, " "))
  scores=c(0,0,0,0,0,0)
  names(scores)=c("anger","fear","joy","love","sadness","surprise")
  for (i in tweet_words[[1]]) {
    k=1
    for (j in words) { 
      inte=intersect(i,names(j))
      if (length(inte)>0) {
        scores[k]=scores[k]+j[which(names(j)==i)]
      }
      k=k+1
    }
  }
  return(names(which.max(scores)))
}

#Save the predicted class vs the real class.
Compare=as.data.frame(test$Sentiment)
for (l in 1:nrow(test)) {
  Compare[l,2]=names(which.max(prediction(test[l,1])))
}

#Return the confusion matrix and the metrics for the model when using the test-set. 
Compare[,1]=as.factor(Compare[,1])
Compare[,2]=as.factor(Compare[,2])
confusionMatrix(Compare[,2],Compare[,1])




