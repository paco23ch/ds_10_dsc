#
# This is the server logic of a Shiny web application. You can run the
# application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
#
#    https://shiny.posit.co/
#

library(shiny)

library(stringi)
library(kableExtra)
library(rJava)
library(RWeka)
library(dplyr)
library(tidyr)
library(tm)
library(caret)

n_gram_limit = 4
file_name <- './train_ngrams_2pct_4grams_deploy.rds'

loading = FALSE

cleanup_input <- function(text) {
  local_corpus <- Corpus(VectorSource(text))
  
  # remove bad words from the sample data set
  #local_corpus <- tm_map(local_corpus, removeWords, badWords)
  #local_corpus <- suppressWarnings(tm_map(local_corpus, removeWords, stopwords("english")))
  local_corpus <- suppressWarnings(tm_map(local_corpus, tolower))
  local_corpus <- suppressWarnings(tm_map(local_corpus, removePunctuation))
  local_corpus <- suppressWarnings(tm_map(local_corpus, removeNumbers))
  local_corpus <- suppressWarnings(tm_map(local_corpus, stripWhitespace))
  
  
  return(content(local_corpus[[1]]))
}

predict_next <- function(input_text, max=3, n_grams, n_gram_limit=4) {
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

# Define server logic required to draw a histogram
function(input, output, session) {
  values <- reactiveValues(n_gram_data = NULL)
  
  observeEvent(input$load_data_btn, {
      output$loaded_data_message <- renderText({"Loading data ... "})
      values$n_gram_data <- readRDS(file=file_name)
      output$loaded_data_message <- renderText({"Data has been loaded."})
    }
  )
  
  calculated_results <- reactive({
    predict_next(input$inputText,input$suggestions, values$n_gram_data, n_gram_limit=input$grams)
  })
  
  output$nextWord <- renderText({
    if(is.null(input$inputText) | input$inputText == '') {
      "*** Data has not yet been loaded ... please press Load Model ***"
    }
    else {
      calculated_results()[[1]]
    }
  })   
  
  output$predictions <- renderTable({
    # Create some sample data for the table
    calculated_results()[2]
  }, digits = 6)

}
