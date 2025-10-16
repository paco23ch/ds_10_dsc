
n_gram_experiment <- n_gram_list[['val']]

for( i in seq_along(n_gram_experiment) ) {
  print(i)
  if(i>1) {
    grams <- c(1:(i-1))
    tokens <- print(paste("token",grams,sep=''))
    n_gram_experiment[[i]] <- n_gram_experiment[[i]] %>% separate('token', into=tokens, remove=FALSE, sep=' ')
    
    for(n in c(1:(i-1))) {
      toks <- c(1:n)
      group = paste('token', toks, sep='')
      col_name = paste('freqt',n, sep='')
      
      print(group)
      print(n)
      
      n_gram_experiment[[i]] <- n_gram_experiment[[i]] %>% group_by(across(all_of(group))) %>% mutate({{col_name}} := sum(freq))
      
      if (i==2) {
        n_gram_experiment[[i]][,'probt1'] = n_gram_experiment[[i]][,'freq'] / n_gram_experiment[[i]][,'freqt1']
      }
      else {
        
      }
      print(col_name) 
    }
    
    if(i==2) {
      n_gram_experiment[[i]][,'probt1'] = n_gram_experiment[[i]][,'freq'] / n_gram_experiment[[i]][,'freqt1']
      n_gram_experiment[[i]][,'probgram'] = n_gram_experiment[[i]]['probt1']
    } else if(i==3) {
      n_gram_experiment[[i]][,'probt1'] = n_gram_experiment[[i]][,'freqt1'] / sum(n_gram_experiment[[i]]$freq)
      n_gram_experiment[[i]][,'probt2'] = n_gram_experiment[[i]][,'freq'] / n_gram_experiment[[i]][,'freqt2']
      n_gram_experiment[[i]][,'probgram'] = n_gram_experiment[[i]]['probt1'] * n_gram_experiment[[i]]['probt2']
    } else if(i==4) {
      n_gram_experiment[[i]][,'probt1'] = n_gram_experiment[[i]][,'freqt1'] / sum(n_gram_experiment[[i]]$freq)
      n_gram_experiment[[i]][,'probt2'] = n_gram_experiment[[i]][,'freqt3'] / n_gram_experiment[[i]][,'freqt2']
      n_gram_experiment[[i]][,'probt3'] = n_gram_experiment[[i]][,'freq'] / n_gram_experiment[[i]][,'freqt3']
      n_gram_experiment[[i]][,'probgram'] = n_gram_experiment[[i]]['probt1'] * n_gram_experiment[[i]]['probt2'] * n_gram_experiment[[i]]['probt3']
    }
  }
}




metrics <- n_gram_experiment[[1]]

metrics <- metrics %>% group_by(token1) %>% mutate(freqt1 = sum(freq))

metrics[,'probt1'] = metrics[,'freq'] / metrics[,'freqt1']

metrics[,'probgram'] = metrics['probt1']

#########

metrics <- n_gram_experiment[[3]]
metrics <- metrics %>% group_by(token1) %>% mutate(freqt1 = sum(freq))
metrics <- metrics %>% group_by(token1, token2) %>% mutate(freqt2 = sum(freq))

metrics[,'probt1'] = metrics[,'freqt1'] / sum(metrics$freq)
metrics[,'probt2'] = metrics[,'freq'] / metrics[,'freqt2']

metrics[,'probgram'] = metrics['probt1'] * metrics['probt2']

#########

metrics <- n_gram_experiment[[4]]

metrics <- metrics %>% group_by(token1) %>% mutate(freqt1 = sum(freq))
metrics <- metrics %>% group_by(token1, token2) %>% mutate(freqt2 = sum(freq))
metrics <- metrics %>% group_by(token1, token2, token3) %>% mutate(freqt3 = sum(freq))

metrics[,'probt1'] = metrics[,'freqt1'] / sum(metrics$freq)
metrics[,'probt2'] = metrics[,'freqt3'] / metrics[,'freqt2']
metrics[,'probt3'] = metrics[,'freq'] / metrics[,'freqt3']

metrics[,'probgram'] = metrics['probt1'] * metrics['probt2'] * metrics['probt3']

##########


metrics <- n_gram_experiment[[4]]

i=4


View(metrics)




View(metrics[!(metrics$word %in% stopwords("english")),])

