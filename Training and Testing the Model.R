options(java.parameters = "-Xmx8192m") # Sets the maximum heap size to 8GB
# You can adjust the value (e.g., 2048m for 2GB, 8192m for 8GB) based on your needs and system resources.

library(stringi)
library(kableExtra)
library(rJava)
library(RWeka)
library(dplyr)
library(tidyr)
library(tm)
library(caret)


# Loading files
read_file <- function(file_name) {
  con <- file(file_name, open = "r")
  data <- readLines(con, encoding = "UTF-8", skipNul = TRUE)
  close(con)
  return(data)
}

calculate_mbytes <- function(bytes) {
  return(round(bytes)/1024 ^ 2)
}

read_data <- function(file_paths) {
  print("Reading data ... ")
  file_contents <- sapply(file_paths, read_file)
  file_size <- sapply(sapply(file_paths, file.info)['size',],calculate_mbytes)
  file_lines <- sapply(file_contents, length)
  file_words <- sapply(file_contents, stri_stats_latex)[4,]
  file_chars <- sapply(sapply(file_contents, nchar),sum)
  file_length <- sapply(file_contents, length)
  file_wpl <- lapply(file_contents, function(x) stri_count_words(x))
  file_wpl_mean <- sapply(file_wpl,mean)
  summary_stats = data.frame(size = file_size, lines = file_lines, 
                             words = file_words, chars = file_chars, words_per_line = file_wpl_mean)
  print(summary_stats)
  return(file_contents)
}

# Sampling
sample_text <- function (data, sample_rate, train, val, test) {
  trainval_pct <- (train)/(train+val)
  rows <- length(data)
  
  sampled_data <- sample(seq_len(rows), size = rows*sample_rate, replace = FALSE)
  subset_data <- data[sampled_data]
  
  inTest <- sample(seq_len(length(subset_data)), size = length(sampled_data)*test, replace = FALSE)
  testData <- subset_data[inTest]
  testData <- iconv(testData, "latin1", "ASCII", sub = "")
  
  trainvalData <- subset_data[-inTest]
  inTrain <- sample(seq_len(length(trainvalData)), size = length(trainvalData)*trainval_pct, replace = FALSE)
  
  trainData <- trainvalData[inTrain]
  trainData <- iconv(trainData, "latin1", "ASCII", sub = "")
  valData <- trainvalData[-inTrain]
  valData <- iconv(valData, "latin1", "ASCII", sub = "")
  
  #print(size*sample_rate*0.8)
  #sampled_data <- sample(sample_data, size * sample_rate * 0.8, replace=FALSE)
  #sampled_data <- iconv(sampled_data, "latin1", "ASCII", sub = "")
  return(list(train=trainData, val=valData, test=testData))
}

sample_data  <- function(file_contents, sample_rate, train, val, test) {
  print("Sampling data ... ")
  file_length <- sapply(file_contents, length)
  sampled_data <- mapply(sample_text, file_contents, sample_rate, train, val, test)
  #sampled_data <- mapply(sample_text, file_contents, file_length, sample_rate=sample_rate)
  
  for(set in c('train','val','test')) {
    sampleDataFileName <- paste('../final_first/en_US/en_US_',sample_rate*100,'_',set,'_sample_data.txt',sep='')
    con <- file(sampleDataFileName, open = 'w')
    for(e in sampled_data[set,]) {
      writeLines(e, con)
    }
    close(con)    
  }
  
  sample_lines <- sapply(sampled_data, length)
  sample_words <- sapply(sapply(sampled_data, stri_count_words), sum)

  sample_summary <- data.frame(lines=sample_lines, words=sample_words)
  sample_summary$words_per_line <- sample_summary$words / sample_summary$lines

  print(sample_summary)
  
  return(sampled_data)
}

# Offensive words
read_offensive <- function(badWordsFileName) {
  print("Reading offensive words ... ")
  con <- file(badWordsFileName, open = "r")
  badWords <- readLines(con)
  badWords <- iconv(badWords, "latin1", "ASCII", sub = "")
  close(con)
  print(paste("Total bad words listed ... ", sum(stri_count_words(badWords))))
  return(badWords)
}

# Corpus
generate_corpus <- function(dataSet, badWords, save_file){
  print("Generating corpus ... ")
  docs <- VCorpus(VectorSource(dataSet))
  toSpace <- content_transformer(function(x, pattern) gsub(pattern, " ", x))
  
  # remove internet formats
  docs <- tm_map(docs, toSpace, "(f|ht)tp(s?)://(.*)[.][a-z]+")
  docs <- tm_map(docs, toSpace, "@[^\\s]+")
  docs <- tm_map(docs, toSpace, "\\b[A-Z a-z 0-9._ - ]*[@](.*?)[.]{1,3} \\b")
  
  # remove bad words from the sample data set
  docs <- tm_map(docs, removeWords, badWords)
  
  # convert to lowercase and remove stop words, punctuation and numbers.
  docs <- tm_map(docs, tolower)
  docs <- tm_map(docs, removePunctuation)
  docs <- tm_map(docs, removeNumbers)
  docs <- tm_map(docs, stripWhitespace)
  #docs <- tm_map(docs, removeWords, stopwords("english"))
  docs <- tm_map(docs, PlainTextDocument)
  
  # save the corpus as a plain text file
  corpusText <- data.frame(text = unlist(sapply(docs, '[', "content")), stringsAsFactors = FALSE)
  con <- file(file_name, open = "w")
  writeLines(corpusText$text, con)
  close(con)
  
  return(docs)
}

split_ngrams <- function(text) {
  words <- strsplit(text, " ")[[1]] # Split by space and extract the vector
  last_word <- words[length(words)] # Get the last element
  remaining_text <- paste(words[-length(words)], collapse = " ")
  return(list(remaining_text, last_word))
}


create_n_grams <- function(corpus_input, n_gram_number=4) {
  n_grams <- list()

  # Unigrams will be the default when no options are found
  for(i in c(1:n_gram_number)) {
    print(paste("*** Generating",i, "grams ... "))
    tokenizer <- function(x) NGramTokenizer(x, Weka_control(min = i, max = i))
    matrix <- TermDocumentMatrix(corpus_input, control = list(tokenize = tokenizer))
    matrixFreq <- sort(rowSums(as.matrix(removeSparseTerms(matrix, 0.99))), decreasing = TRUE)
    rm(matrix)
    matrixFreq <- data.frame(word = names(matrixFreq), freq = matrixFreq)

    if(i>1 & dim(matrixFreq)[1]>0) {
      matrixFreq <- matrixFreq %>% separate(word, into = c("token","word"), sep = " (?=[^ ]*$)")
      
      grams <- c(1:(i-1))
      tokens <- paste("token",grams,sep='')
      matrixFreq <- matrixFreq %>% separate('token', into=tokens, remove=FALSE, sep=' ')
      
      for(n in c(1:(i-1))) {
        toks <- c(1:n)
        group = paste('token', toks, sep='')
        col_name = paste('freqt',n, sep='')
        matrixFreq <- matrixFreq %>% group_by(across(all_of(group))) %>% mutate({{col_name}} := sum(freq))
        
      }
    } else {
      matrixFreq <- matrixFreq[!(matrixFreq$word %in% stopwords("english")),]
    }
    
    print(paste("*** Total entries ",dim(matrixFreq)[1]," ...", sep=''))
    
    print("*** Calculating probabilities ... ")
    if(i==1) {
      matrixFreq[,'probgram'] = matrixFreq[,'freq'] / sum(matrixFreq$freq)
    } else if(i==2) {
      matrixFreq[,'probt1'] = matrixFreq[,'freq'] / matrixFreq[,'freqt1']
      matrixFreq[,'probgram'] = matrixFreq['probt1']
    } else if(i==3) {
      matrixFreq[,'probt1'] = matrixFreq[,'freqt1'] / sum(matrixFreq$freq)
      matrixFreq[,'probt2'] = matrixFreq[,'freq'] / matrixFreq[,'freqt2']
      matrixFreq[,'probgram'] = matrixFreq['probt1'] * matrixFreq['probt2']
    } else if(i==4) {
      matrixFreq[,'probt1'] = matrixFreq[,'freqt1'] / sum(matrixFreq$freq)
      matrixFreq[,'probt2'] = matrixFreq[,'freqt3'] / matrixFreq[,'freqt2']
      matrixFreq[,'probt3'] = matrixFreq[,'freq'] / matrixFreq[,'freqt3']
      matrixFreq[,'probgram'] = matrixFreq['probt1'] * matrixFreq['probt2'] * matrixFreq['probt3']
    }
    
    n_grams <- c(n_grams, list(matrixFreq))
  }
  
  return(n_grams)
}

cleanup_input <- function(text) {
  local_corpus <- Corpus(VectorSource(text))
  
  # remove bad words from the sample data set
  local_corpus <- tm_map(local_corpus, removeWords, badWords)
  #local_corpus <- suppressWarnings(tm_map(local_corpus, removeWords, stopwords("english")))
  local_corpus <- suppressWarnings(tm_map(local_corpus, tolower))
  local_corpus <- suppressWarnings(tm_map(local_corpus, removePunctuation))
  local_corpus <- suppressWarnings(tm_map(local_corpus, removeNumbers))
  local_corpus <- suppressWarnings(tm_map(local_corpus, stripWhitespace))
  

  return(content(local_corpus[[1]]))
}

predict_next <- function(input_text, max=3, n_grams, n_gram_limit=4) {
  print(input_text)
  input_text <- cleanup_input(input_text)
  print(input_text)
  words_vector <- unlist(strsplit(input_text," "))
  preds <- NULL
  schema <- NULL
  
  if(is.null(input_text) | input_text == '') {
    print("Predicting from null...")
    schema <- n_grams[[1]]
  } else {
    potential_words <- length(words_vector)
    use_ngram <- potential_words + 1
    if(potential_words > n_gram_limit-1) {
      use_ngram <- n_gram_limit
      words_vector <- tail(words_vector, n_gram_limit-1)
      input_text <- paste(words_vector, collapse = " ")
    } 
    schema = n_grams[[use_ngram]]
  } 

  for(i in c(n_gram_limit:1)) {
    print('Searching ...')
    this_ngram <- NULL
    search_vector <- tail(words_vector, i-1)
    print(i)
    search_text <- paste(search_vector, collapse = " ")
    print(search_text)
    if(i>1) {
      this_ngram <- n_grams[[i]][n_grams[[i]]['token']==search_text,]
    } else {
      this_ngram <- head(n_grams[[i]],max)
    }
    
    if(dim(this_ngram)[1] > 0) {
      preds <- head(this_ngram,max)
      return(preds)
    }

  }
  return(preds)
}

set.seed(2222)

file_paths <- c("../final_first/en_US/en_US.blogs.txt", "../final_first/en_US/en_US.twitter.txt", "../final_first/en_US/en_US.news.txt")
badWordsFileName <- "../final_first/en_US/en.txt"
n_gram_limit = 4
sample_rate = 0.02
trainpct <- 0.8
valpct <- 0.1
testpct <- 0.1
trainval_pct <- (trainpct)/(trainpct+valpct)

file_contents <- read_data(file_paths = file_paths)
badWords <- read_offensive(badWordsFileName)


print(paste('Sampling',sample_rate*100,'percent'))
sampled_data <- sample_data(file_contents, sample_rate = sample_rate, trainpct, valpct, testpct)

# rm(file_contents)

#
print('Corpus ...')
corpusList=list()
for(set in c('train','val','test')) {
  print(paste('Generating corpus for',set,'...',sep=' '))
  file_name <- paste('../final_first/en_US/',set,'_corpus_',sample_rate*100,'pct.txt',sep='')
  corpusSet <- generate_corpus(sampled_data[set,], badWords, file_name)
  file_name <- paste('../final_first/en_US/',set,'_corpus_',sample_rate*100,'pct.rds',sep='')
  print(paste('Saving',set,'...',sep=' '))
  saveRDS(corpusSet, file=file_name)
  corpusList[set] = list(corpusSet)
}

#rm(sampled_data)
print('n-grams...')
n_gram_list = list()
for(set in c('train','val','test')) {
  print(paste('Generating n-grams for',set,'...',sep=' '))
  file_name <- paste('../final_first/en_US/',set,'_ngrams_',sample_rate*100,'pct_',n_gram_limit,'grams.rds',sep='')
  n_grams_set <- create_n_grams(corpusList[[set]], n_gram_number=n_gram_limit)
  print(paste('Saving',set,'...',sep=' '))
  saveRDS(n_grams_set, file=file_name)
  n_gram_list[set] = list(n_grams_set)
}

#predict_next("", 1, n_grams) 

#Load n-grams
n_gram_list = list()
n_gram_limit = 4
sample_rate = 0.02
set = 'train'
file_name <- paste('../final_first/en_US/',set,'_ngrams_',sample_rate*100,'pct_',n_gram_limit,'grams.rds',sep='')
n_grams_set <- readRDS(file=file_name)
n_gram_list[set] = list(n_grams_set)


predict_next("blabla", 1, n_gram_list[['train']], n_gram_limit=n_gram_limit)
predict_next("nonsense", 1, n_gram_list[['train']], n_gram_limit=n_gram_limit)
predict_next("right", 1, n_gram_list[['train']], n_gram_limit=n_gram_limit)
predict_next("let us", 1, n_gram_list[['train']], n_gram_limit=n_gram_limit)
predict_next("martin luther king", 1, n_gram_list[['train']], n_gram_limit=n_gram_limit)
predict_next("what the hell are you",1, n_gram_list[['train']], n_gram_limit=n_gram_limit)

predict_next("The guy in front of me just bought a pound of bacon, a bouquet, and a case of",3, n_gram_list[['train']], n_gram_limit=n_gram_limit)
predict_next("You're the reason why I smile everyday. Can you follow me please? It would mean the",3, n_gram_list[['train']], n_gram_limit=n_gram_limit)
predict_next("Hey sunshine, can you follow me and make me the",3, n_gram_list[['train']], n_gram_limit=n_gram_limit)
predict_next("Very early observations on the Bills game: Offense still struggling but the",3, n_gram_list[['train']], n_gram_limit=n_gram_limit)
predict_next("Go on a romantic date at the",3, n_gram_list[['train']], n_gram_limit=n_gram_limit)
predict_next("Well I'm pretty sure my granny has some old bagpipes in her garage I'll dust them off and be on my",3, n_gram_list[['train']], n_gram_limit=n_gram_limit)
predict_next("Ohhhhh #PointBreak is on tomorrow. Love that film and haven't seen it in quite some",3, n_gram_list[['train']], n_gram_limit=n_gram_limit)
predict_next("After the ice bucket challenge Louis will push his long wet hair out of his eyes with his little",3, n_gram_list[['train']], n_gram_limit=n_gram_limit)
predict_next("Be grateful for the good times and keep the faith during the",3, n_gram_list[['train']], n_gram_limit=n_gram_limit)
predict_next("If this isn't the cutest thing you've ever seen, then you must be",3, n_gram_list[['train']], n_gram_limit=n_gram_limit)


predict_next("When you breathe, I want to be the air for you. I'll be there for you, I'd live and I'd",3, n_gram_list[['train']], n_gram_limit=n_gram_limit)
predict_next("Guy at my table's wife got up to go to the bathroom and I asked about dessert and he started telling me about his",3, n_gram_list[['train']], n_gram_limit=n_gram_limit)
predict_next("I'd give anything to see arctic monkeys this",3, n_gram_list[['train']], n_gram_limit=n_gram_limit)
predict_next("Talking to your mom has the same effect as a hug and helps reduce your",3, n_gram_list[['train']], n_gram_limit=n_gram_limit)
predict_next("When you were in Holland you were like 1 inch away from me but you hadn't time to take a",3, n_gram_list[['train']], n_gram_limit=n_gram_limit)
predict_next("I'd just like all of these questions answered, a presentation of evidence, and a jury to settle the",3, n_gram_list[['train']], n_gram_limit=n_gram_limit)
predict_next("I can't deal with unsymetrical things. I can't even hold an uneven number of bags of groceries in each",3, n_gram_list[['train']], n_gram_limit=n_gram_limit)
predict_next("Every inch of you is perfect from the bottom to the",3, n_gram_list[['train']], n_gram_limit=n_gram_limit)
predict_next("I’m thankful my childhood was filled with imagination and bruises from playing",3, n_gram_list[['train']], n_gram_limit=n_gram_limit)
predict_next("I like how the same people are in almost all of Adam Sandler's",3, n_gram_list[['train']], n_gram_limit=n_gram_limit)

questions = c("The guy in front of me just bought a pound of bacon, a bouquet, and a case of",
              "You're the reason why I smile everyday. Can you follow me please? It would mean the",
              "Hey sunshine, can you follow me and make me the",
              "Very early observations on the Bills game: Offense still struggling but the",
              "Go on a romantic date at the",
              "Well I'm pretty sure my granny has some old bagpipes in her garage I'll dust them off and be on my",
              "Ohhhhh #PointBreak is on tomorrow. Love that film and haven't seen it in quite some",
              "After the ice bucket challenge Louis will push his long wet hair out of his eyes with his little",
              "Be grateful for the good times and keep the faith during the",
              "If this isn't the cutest thing you've ever seen, then you must be",
              "When you breathe, I want to be the air for you. I'll be there for you, I'd live and I'd",
              "Guy at my table's wife got up to go to the bathroom and I asked about dessert and he started telling me about his",
              "I'd give anything to see arctic monkeys this",
              "Talking to your mom has the same effect as a hug and helps reduce your",
              "When you were in Holland you were like 1 inch away from me but you hadn't time to take a",
              "I'd just like all of these questions answered, a presentation of evidence, and a jury to settle the",
              "I can't deal with unsymetrical things. I can't even hold an uneven number of bags of groceries in each",
              "Every inch of you is perfect from the bottom to the",
              "I’m thankful my childhood was filled with imagination and bruises from playing",
              "I like how the same people are in almost all of Adam Sandler's")

answers = c("beer", "world","happiest","defense","beach","way","time","finger","bad","insane",
  "die","marital","weekend","stess","picture","matter","hand","top","outside","movies")

files = c("../final_first/en_US/train_ngrams_2pct_4grams.rds",
          "../final_first/en_US/train_ngrams_5pct_4grams.rds",
          "../final_first/en_US/train_ngrams_15pct_4grams.rds",
          "../final_first/en_US/train_ngrams_25pct_4grams.rds",
          "../final_first/en_US/train_ngrams_61pct_4grams.rds"
          )

percents = c(2,5,15,25,61)
n_grams_list = c(2,3,4)


predict_next_app <- function(input_text, max=3, n_grams, n_gram_limit=4) {
  input_text <- cleanup_input(input_text)
  words_vector <- unlist(strsplit(input_text," "))
  preds <- NULL
  schema <- NULL
  
  if(is.null(input_text) | input_text == '') {
    schema <- n_grams[[1]]
  } else {
    potential_words <- length(words_vector)
    use_ngram <- potential_words + 1
    if(potential_words > n_gram_limit-1) {
      use_ngram <- n_gram_limit
      words_vector <- tail(words_vector, n_gram_limit-1)
      input_text <- paste(words_vector, collapse = " ")
    } 
    schema = n_grams[[use_ngram]]
  } 
  
  for(i in c(n_gram_limit:1)) {
    this_ngram <- NULL
    search_vector <- tail(words_vector, i-1)
    search_text <- paste(search_vector, collapse = " ")
    if(i>1) {
      this_ngram <- n_grams[[i]][n_grams[[i]]['token']==search_text,]
    } else {
      this_ngram <- head(n_grams[[i]],max)
    }
    
    if(!is.null(this_ngram)) {
      if(dim(this_ngram)[1] > 0) {
        preds <- head(this_ngram,max)
        return(list(head(preds,1)$word, preds))
      }      
    } else
      return(list(NULL,NULL))
    
  }
  return( list(head(preds,1)$word, preds))
}

pct_array = c()
ngl_array = c()
q_array = c()
true_y = c()
pred_y = c()

sum_pct_array = c()
sum_ngl_array = c()
acc_array = c()

for(file in seq_along(files)) {
  print(paste("reading",files[file]))
  n_grams_set <- readRDS(file=files[file])
  for(ng_limit in n_grams_list) {
    print(paste("ng_limit",ng_limit))
    correct = 0
    for(i in seq_along(questions)) {
      results = predict_next_app(questions[i],3, n_grams_set, n_gram_limit=ng_limit)
      if(results[[1]] == answers[i]) {
        correct = correct + 1
      }
      pct_array = c(pct_array, percents[file])
      ngl_array = c(ngl_array, ng_limit)
      q_array = c(q_array, i)
      true_y = c(true_y, answers[i])
      pred_y = c(pred_y, results[[1]])
    }
    accuracy = correct / length(questions)
    sum_pct_array = c(sum_pct_array, percents[file])
    sum_ngl_array = c(sum_ngl_array, ng_limit)
    acc_array = c(acc_array, accuracy)
  }
}

results <- data.frame(percent=pct_array, n_grams=ngl_array, question=q_array, y=true_y, y_hat=pred_y)

summary <- data.frame(percent=sum_pct_array, n_grams=sum_ngl_array, accuracy=acc_array)
  
file_name <- '../final_first/en_US/train_ngrams_25pct_4grams.rds'
n_grams_set <- readRDS(file=file_name)
new_file_name <- '../final_first/en_US/train_ngrams_25pct_4grams_deploy.rds'
for(i in seq_along(n_grams_set)) {
  if(i>1) {
    n_grams_set[[i]] <- n_grams_set[[i]][,c('token','word','freq','probgram')]
  }
  
}
saveRDS(n_grams_set, new_file_name)
  
  