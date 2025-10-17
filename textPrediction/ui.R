#
# This is the user-interface definition of a Shiny web application. You can
# run the application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
#
#    https://shiny.posit.co/
#

library(shiny)
library(bslib)

# Define UI for application that draws a histogram
fluidPage(

    # Application title
    titlePanel("Inspiration corner"),

    # Sidebar with a slider input for number of bins
    sidebarLayout(
        sidebarPanel(
          h2("Give us a start"),
          p("In this side, you can start typing a few words and have our engine give you some suggestions."),
          p("You can select the number of words to use to predict the next word, plus narrow down the number of additional recommendations to the most likely candidate."),
            textInput(inputId = 'inputText', label = "Please start typing to get some inspiration ... ", value=''),
            sliderInput("grams",
                        "Number n-grams to use:",
                        min = 2,
                        max = 4,
                        value = 3),
            sliderInput("suggestions",
                        "Number suggestions to generate",
                        min = 1,
                        max = 10,
                        value = 4)
        ),

        # Show a plot of the generated distribution
        mainPanel(
          h1("Instructions"),
          p("Since the application uses a large set of data to work, it will need to be loaded manually, so please:"),
          tags$li("Click on the Load Model button"),
          tags$li("Wait for the Loaded Data message to appear, and"),
          tags$li("Start typing on the left panel"),
          actionButton("load_data_btn", "Load Model"),
          textOutput(outputId = "loaded_data_message"),
          h2("Our best guesses ..."),
          p("On this side, you'll see the suggestions come alive, starting with the most likely guess, plus a few more suggestions and their likelihoods."),
          p("This version of the model has been reduced to a 5% of sampling of the total corpus for portability, and we've tested that with larger sample we get beter results, so as a proof-of-concept this will be a very good way to get inspiration."),
          fluidRow(
            column(12, offset = , h3("Most likely word")),
            textOutput(outputId = "nextWord")# Uses offset for spacing
          ),
          
          fluidRow(
            column(12, offset = 0, h3("Other suggestions")),
            tableOutput(outputId = "predictions")# Uses offset for spacing
          )
          
        )
    )
)
