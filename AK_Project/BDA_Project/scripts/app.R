library(shiny)
library(shinydashboard)
library(ggplot2)
library(DT)
library(dplyr)

# UI
ui <- dashboardPage(
  dashboardHeader(title = "Student Retention EWS"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Dashboard", tabName = "dashboard", icon = icon("dashboard")),
      menuItem("High Risk Students", tabName = "high_risk", icon = icon("exclamation-triangle"))
    )
  ),
  dashboardBody(
    tabItems(
      tabItem(tabName = "dashboard",
        fluidRow(
          valueBoxOutput("totalStudents"),
          valueBoxOutput("atRiskCount"),
          valueBoxOutput("avgEngagement")
        ),
        fluidRow(
          box(title = "Risk Distribution", status = "primary", solidHeader = TRUE,
              plotOutput("riskDistPlot")),
          box(title = "Engagement vs Risk", status = "primary", solidHeader = TRUE,
              plotOutput("engagementPlot"))
        )
      ),
      tabItem(tabName = "high_risk",
        fluidRow(
          box(title = "High Risk Intervention List", width = 12,
              DTOutput("riskTable"))
        ),
        fluidRow(
          box(title = "Student Profile View", width = 12,
              uiOutput("studentProfile"))
        )
      )
    )
  )
)

# Server
server <- function(input, output) {

  # Load Data
  data <- reactive({
    # In a real app, this might connect to Hive or a Database
    read.csv("/home/data/full_risk_report.csv")
  })

  output$totalStudents <- renderValueBox({
    valueBox(nrow(data()), "Total Students", icon = icon("users"), color = "blue")
  })

  output$atRiskCount <- renderValueBox({
    at_risk <- data() %>% filter(risk_level == "High Risk") %>% nrow()
    valueBox(at_risk, "High Risk Students", icon = icon("warning"), color = "red")
  })

  output$avgEngagement <- renderValueBox({
    avg_logins <- mean(data()$total_logins, na.rm = TRUE)
    valueBox(round(avg_logins, 1), "Avg Logins", icon = icon("mouse-pointer"), color = "green")
  })

  output$riskDistPlot <- renderPlot({
    ggplot(data(), aes(x = risk_level, fill = risk_level)) +
      geom_bar() +
      theme_minimal() +
      labs(x = "Risk Level", y = "Count")
  })

  output$engagementPlot <- renderPlot({
    ggplot(data(), aes(x = total_logins, y = risk_score)) +
      geom_point(alpha = 0.5, aes(color = risk_level)) +
      geom_smooth(method = "lm") +
      theme_minimal()
  })

  output$riskTable <- renderDT({
    data() %>%
      filter(risk_level == "High Risk") %>%
      select(student_id, historical_gpa, total_logins, risk_score) %>%
      datatable(selection = 'single')
  })

  output$studentProfile <- renderUI({
    s <- input$riskTable_rows_selected
    if (length(s)) {
      student <- data()[s, ]
      tagList(
        h3(paste("Profile for:", student$student_id)),
        p(paste("Risk Probability:", round(student$risk_score, 4))),
        actionButton("sendAlert", "Send Counselor Alert", class = "btn-danger")
      )
    } else {
      p("Select a student from the table to view details.")
    }
  })

  observeEvent(input$sendAlert, {
    showModal(modalDialog(
      title = "Alert Sent",
      "Counselor has been notified via email system simulation.",
      easyClose = TRUE
    ))
  })
}

shinyApp(ui, server)
