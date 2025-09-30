library(stringi)
library(kableExtra)
library(rJava)
library(RWeka)
library(dplyr)
library(tidyr)
library(tm)


sample_rate = 0.2
set.seed(2222)
file_paths <- c("../final_first/en_US/en_US.blogs.txt", "../final_first/en_US/en_US.twitter.txt", "../final_first/en_US/en_US.news.txt")
badWordsFileName <- "../final_first/en_US/en.txt"

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
sample_text <- function (data, size, sample_rate) {
  sampled_data <- sample(data, size * 0.5, replace=FALSE)
  sampled_data <- iconv(sampled_data, "latin1", "ASCII", sub = "")
  return(sampled_data)
}

sample_data  <- function(file_contents, sample_rate) {
  print("Sampling data ... ")
  file_length <- sapply(file_contents, length)
  sampled_data <- mapply(sample_text, file_contents, file_length, sample_rate=sample_rate)
  
  sampleDataFileName <- '../final_first/en_US/en_US.sample_data.txt'
  con <- file(sampleDataFileName, open = 'w')

  for(e in sampled_data) {
    writeLines(e, con)
  }
  close(con)

  sample_lines <- sapply(sampled_data, length)
  sample_words <- sapply(sapply(sampled_data, stri_count_words), sum)

  sample_summary <- data.frame(lines=sample_lines, words=sample_words)
  sample_summary$words_per_line <- sample_summary$words / sample_summary$lines

  # get number of lines and words from the sample data set
  sampleDataLines <- sum(sapply(sampled_data,length))
  print(paste('Total sampled lines: ',sampleDataLines))

  sampleDataWords <- sum(stri_count_words(sampled_data))
  print(paste('Total Sampled words: ', sampleDataWords))
  
  
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
generate_corpus <- function(dataSet, badWords){
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
  docs <- tm_map(docs, removeWords, stopwords("english"))
  docs <- tm_map(docs, removePunctuation)
  docs <- tm_map(docs, removeNumbers)
  docs <- tm_map(docs, stripWhitespace)
  docs <- tm_map(docs, PlainTextDocument)
  
  # save the corpus file for later use
  saveRDS(docs, file = "../final_first/en_US/en_US.corpus.rds")
  
  # save the corpus as a plain text file
  corpusText <- data.frame(text = unlist(sapply(docs, '[', "content")), stringsAsFactors = FALSE)
  con <- file("../final_first/en_US/en_US.corpus.txt", open = "w")
  writeLines(corpusText$text, con)
  close(con)
  
  return(docs)
}

create_n_grams <- function(corpus) {
  print("Setting up n-grams ... ")
  
  # n-gram generation
  unigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 1))
  bigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))
  trigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 3, max = 3))
  fourgramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 4, max = 4))
  
  # Unigrams will be the default when no options are found
  print("Generating Unigrams ... ")
  unigramMatrix <- TermDocumentMatrix(corpus, control = list(tokenize = unigramTokenizer))
  unigramMatrixFreq <- sort(rowSums(as.matrix(removeSparseTerms(unigramMatrix, 0.99))), decreasing = TRUE)
  unigramMatrixFreq <- data.frame(word = names(unigramMatrixFreq), freq = unigramMatrixFreq)
  print(paste("Dimensions:",dim(unigramMatrixFreq)))
  
  # Creating the bigram model
  print("Generating bigram ... ")
  bigramMatrix <- TermDocumentMatrix(corpus, control = list(tokenize = bigramTokenizer))
  bigramMatrixFreq <- sort(rowSums(as.matrix(removeSparseTerms(bigramMatrix, 0.99))), decreasing = TRUE)
  bigramMatrixFreq <- data.frame(word = names(bigramMatrixFreq), freq = bigramMatrixFreq)
  bigramMatrixFreq <- bigramMatrixFreq %>% separate(word, into = c("token1","word"), sep=" ") %>% 
    arrange(token1, desc(freq))
  print(paste("Dimensions:",dim(bigramMatrixFreq)))
  
  # Creating the trigram model
  print("Generating trigram ... ")
  trigramMatrix <- TermDocumentMatrix(corpus, control = list(tokenize = trigramTokenizer))
  trigramMatrixFreq <- sort(rowSums(as.matrix(removeSparseTerms(trigramMatrix, 0.99))), decreasing = TRUE)
  trigramMatrixFreq <- data.frame(word = names(trigramMatrixFreq), freq = trigramMatrixFreq)
  trigramMatrixFreq <- trigramMatrixFreq %>% separate(word, into = c("token1","token2","word"), sep=" ") %>%
    arrange(token1, token2, desc(freq))
  print(paste("Dimensions:",dim(trigramMatrixFreq)))
  
  # Creating the fourgram model
  print("Generating fourgram ... ")
  fourgramMatrix <- TermDocumentMatrix(corpus, control = list(tokenize = fourgramTokenizer))
  fourgramMatrixFreq <- sort(rowSums(as.matrix(removeSparseTerms(fourgramMatrix, 0.99))), decreasing = TRUE)
  fourgramMatrixFreq <- data.frame(word = names(fourgramMatrixFreq), freq = fourgramMatrixFreq)
  fourgramMatrixFreq <- fourgramMatrixFreq %>% separate(word, into = c("token1","token2","token3","word"), sep=" ") %>%
    arrange(token1, token2, token3, desc(freq))
  print(paste("Dimensions:",dim(fourgramMatrixFreq)))
  
  return(list(unigramMatrixFreq,bigramMatrixFreq, trigramMatrixFreq, fourgramMatrixFreq))
}


cleanup_input <- function(text) {
  local_corpus <- Corpus(VectorSource(text))
  
  # remove bad words from the sample data set
  #local_corpus <- tm_map(local_corpus, removeWords, badWords)

  local_corpus <- tm_map(local_corpus, tolower)
  local_corpus <- tm_map(local_corpus, removePunctuation)
  local_corpus <- tm_map(local_corpus, removeNumbers)
  local_corpus <- tm_map(local_corpus, stripWhitespace)

  return(content(local_corpus[[1]]))
}

# Generate Predictions
predict_next <- function(input_text, max=3, n_grams, n_gram_limit=4) {
  input_text <- cleanup_input(input_text)
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
      words_vector <- tail(words_vector, n_gram_limit - 1)
    } 
    schema = n_grams[[use_ngram]]
  } 
  
  preds <- schema
  for(i in seq_along(words_vector)) {
    token <- paste("token",i,sep='')
    preds <- preds[preds[token]==words_vector[i],]
  }
    
  if(dim(preds)[1] == 0) {
    preds <- n_grams[[1]]
  }
  
  preds$prob = preds$freq / sum(preds$freq)
  return(head(preds, max)[,c('word','prob')])
}




file_contents <- read_data(file_paths = file_paths)
sampled_data <- sample_data(file_contents, 1)
badWords <- read_offensive(badWordsFileName)
corpus <- generate_corpus(dataSet, badWords)
n_grams <- create_n_grams(corpus)


predict_next("", 1, n_grams)
predict_next("right", 1, n_grams)
predict_next("let us", 1, n_grams)
predict_next("martin luther king", 1, n_grams)
predict_next("what the hell are you",1, n_grams)

predict_next("what, the, fuck_ are you?",1, n_grams)


input_text <- "this is a"
words_vector <- unlist(strsplit(input_text," "))

