# Load Libraries
library(tidyverse)
library(caret)
library(pROC)

# 1. Load Data
# Assuming columns: student_id, historical_gpa, total_logins, avg_delay_days, video_hours, forum_activity_score, label_dropped
data <- read.csv("/home/data/export/student_features.csv", header=FALSE)
colnames(data) <- c("student_id", "historical_gpa", "total_logins", "avg_delay_days", "video_hours", "forum_activity_score", "label_dropped")

# Handle NAs (e.g., students with no activity)
data[is.na(data)] <- 0

# 2. Preprocessing
# Convert label to factor
data$label_dropped <- as.factor(data$label_dropped)

# Split Data
set.seed(123)
trainIndex <- createDataPartition(data$label_dropped, p = .8, list = FALSE)
trainData <- data[trainIndex,]
testData  <- data[-trainIndex,]

# 3. Train Logistic Regression Model
model <- glm(label_dropped ~ historical_gpa + total_logins + avg_delay_days + video_hours + forum_activity_score,
             data = trainData, family = "binomial")

summary(model)

# 4. Evaluation
probs <- predict(model, newdata = testData, type = "response")
preds <- ifelse(probs > 0.5, 1, 0)

# Confusion Matrix
conf_matrix <- confusionMatrix(as.factor(preds), testData$label_dropped)
print(conf_matrix)

# ROC-AUC
roc_obj <- roc(testData$label_dropped, probs)
print(paste("AUC:", auc(roc_obj)))

# 5. Risk Scoring for all students
data$risk_score <- predict(model, newdata = data, type = "response")

# Categorize Risk
data <- data %>%
  mutate(risk_level = case_when(
    risk_score < 0.35 ~ "Low Risk",
    risk_score >= 0.35 & risk_score < 0.70 ~ "Moderate Risk",
    risk_score >= 0.70 ~ "High Risk"
  ))

# 6. Export High Risk Interventions
high_risk <- data %>% filter(risk_level == "High Risk")
write.csv(high_risk, "/home/data/high_risk_interventions.csv", row.names = FALSE)
write.csv(data, "/home/data/full_risk_report.csv", row.names = FALSE)

print("Retention Prediction Complete. Files exported to /home/data/")
