# Install.packages
library(plyr)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(caret)
library(MASS)

# Import CSV to a dataframe
medical <- read.csv("~/MSDA/D208 Predictive Modeling/D208 performance assessment task2/medical_clean.csv")

# View the structure of the data
str(medical)

# Detect duplicates
sum(duplicated(medical))

# Detect missing values
colSums(is.na(medical))

# Detect outlier values
# Quantitative outliers
quantitative_outliers <- function(dataframe, column_name) {
  # Extract the specified column
  column_data <- dataframe[[column_name]]
  
  # Calculate the mean and standard deviation of the column
  column_mean <- mean(column_data, na.rm = TRUE)
  column_sd <- sd(column_data, na.rm = TRUE)
  
  # Calculate the z-scores for each value in the column
  z_scores <- (column_data - column_mean) / column_sd
  
  # Count the number of outliers (z-scores > 3 or < -3)
  num_outliers <- sum(abs(z_scores) > 3, na.rm = TRUE)
  
  # Print the column name and outlier count
  cat("Variable", column_name, "has", num_outliers, "outliers.\n")
  
  # Return the z-scores and the number of outliers
  result <- list(z_scores = z_scores, num_outliers = num_outliers)
  return(result)
}
# Check the quantitative variables for outliers
result <- quantitative_outliers(medical, "Population")
result <- quantitative_outliers(medical, "Children")
result <- quantitative_outliers(medical, "Age")
result <- quantitative_outliers(medical, "Income")
result <- quantitative_outliers(medical, "VitD_levels")
result <- quantitative_outliers(medical, "Doc_visits")
result <- quantitative_outliers(medical, "Full_meals_eaten")
result <- quantitative_outliers(medical, "vitD_supp")
result <- quantitative_outliers(medical, "Initial_days")
result <- quantitative_outliers(medical, "TotalCharge")
result <- quantitative_outliers(medical, "Additional_charges")

# Detect unique values in categorical variables to check for inconsistencies in data presentation
# Specify the dataframe and column names
df <- medical
column_names <- c("Job", "Marital", "Gender", "ReAdmis", "Soft_drink", "Initial_admin", "HighBlood", "Stroke", "Complication_risk", "Overweight", "Arthritis", "Diabetes", "Hyperlipidemia", "BackPain", "Anxiety", "Allergic_rhinitis", "Reflux_esophagitis", "Asthma", "Services", "Item1", "Item2", "Item3", "Item4", "Item5", "Item6", "Item7", "Item8")

# Detect and print unique values in the specified columns
for (column_name in column_names) {
  unique_values <- unique(df[[column_name]])
  cat("Unique values in column", column_name, ":\n")
  print(unique_values)
  cat("\n")
}


# Summary statistics for independent quantitative variables
summary(medical$Children)
summary(medical$Age)
summary(medical$Income)
summary(medical$VitD_levels)
summary(medical$Doc_visits)
summary(medical$Full_meals_eaten)
summary(medical$vitD_supp)
summary(medical$Initial_days)
summary(medical$TotalCharge)
summary(medical$Additional_charges)

str(medical)


# Re-express categorical variables
# One-hot encoding
# (Zach, 2021a)
dmy <- dummyVars(" ~ Gender", data = medical)
trsf <- data.frame(predict(dmy, newdata = medical))
str(trsf)

# Combine the encoded data with the original dataframe
medical_encoded <- cbind(medical, trsf)

# view updated dataframe
str(medical_encoded)

# Label encoding of categorical variables with two unique values
# (Zach, 2021)
# Specify the columns to be updated
columns_to_update <- c("ReAdmis", "HighBlood", "Stroke", "Overweight", "Arthritis", "Diabetes", "Hyperlipidemia", "BackPain", "Anxiety", "Allergic_rhinitis", "Reflux_esophagitis", "Asthma")

# Use lapply to apply revalue to the specified columns
medical_encoded[columns_to_update] <- lapply(medical_encoded[columns_to_update], function(x) revalue(x, c("Yes" = 1, "No" = 0)))
# convert char categorical variables to numeric
medical_encoded$ReAdmis <- as.numeric(medical_encoded$ReAdmis)
medical_encoded$HighBlood <- as.numeric(medical_encoded$HighBlood)
medical_encoded$Stroke <- as.numeric(medical_encoded$Stroke)
medical_encoded$Overweight <- as.numeric(medical_encoded$Overweight)
medical_encoded$Arthritis <- as.numeric(medical_encoded$Arthritis)
medical_encoded$Diabetes <- as.numeric(medical_encoded$Diabetes)
medical_encoded$Hyperlipidemia <- as.numeric(medical_encoded$Hyperlipidemia)
medical_encoded$BackPain <- as.numeric(medical_encoded$BackPain)
medical_encoded$Anxiety <- as.numeric(medical_encoded$Anxiety)
medical_encoded$Allergic_rhinitis <- as.numeric(medical_encoded$Allergic_rhinitis)
medical_encoded$Reflux_esophagitis <- as.numeric(medical_encoded$Reflux_esophagitis)
medical_encoded$Asthma <- as.numeric(medical_encoded$Asthma)

# Display the updated dataframe
str(medical_encoded)

# Save prepared dataset to CSV
# Get the current working directory
current_dir <- getwd()

# Specify the CSV file name in the current directory
file_path <- file.path(current_dir, "medical_prepared.csv")

# Save the dataframe to a CSV file in the current directory
write.csv(medical_encoded, file = file_path, row.names = FALSE)

# Print a message to confirm the save operation
cat("Dataframe saved to", file_path, "\n")


# Initial regression model
logres_initial <- glm(formula = ReAdmis ~ Age + GenderFemale + GenderMale + VitD_levels + HighBlood + Stroke + Overweight + Arthritis + Diabetes + Hyperlipidemia + BackPain + Anxiety + Allergic_rhinitis + Reflux_esophagitis + Asthma + Children + Income + Doc_visits + Full_meals_eaten + vitD_supp + Initial_days + TotalCharge + Additional_charges, data = medical_encoded, family = binomial)

summary(logres_initial)

# Backward stepwise elimination
# Function to display variables with p-values greater than 0.05 in descending order by p-value
display_high_p_value_variables <- function(model) {
  p_values <- summary(model)$coefficients[, "Pr(>|z|)"]
  high_p_value_variables <- names(p_values[p_values > 0.05])
  
  if (length(high_p_value_variables) > 0) {
    cat("Variables with p-values greater than 0.05 (in descending order by p-value):\n")
    
    # Sort the variables by p-value in descending order
    sorted_high_p_value_variables <- high_p_value_variables[order(-p_values[high_p_value_variables])]
    
    for (variable in sorted_high_p_value_variables) {
      p_value <- p_values[variable]
      cat("Variable:", variable, "with p-value:", p_value, "\n")
    }
  } else {
    cat("No variables with p-values greater than 0.05\n")
  }
}

display_high_p_value_variables(logres_initial)

# step 2 - remove Doc_visits variable and rerun model
logres_step2 <- glm(formula = ReAdmis ~ Age + GenderFemale + GenderMale + VitD_levels + HighBlood + Stroke + Overweight + Arthritis + Diabetes + Hyperlipidemia + BackPain + Anxiety + Allergic_rhinitis + Reflux_esophagitis + Asthma + Children + Income + Full_meals_eaten + vitD_supp + Initial_days + TotalCharge + Additional_charges, data = medical_encoded, family = binomial)

summary(logres_step2)
display_high_p_value_variables(logres_step2)

# step 3 - remove Income variable and rerun model
logres_step3 <- glm(formula = ReAdmis ~ Age + GenderFemale + GenderMale + VitD_levels + HighBlood + Stroke + Overweight + Arthritis + Diabetes + Hyperlipidemia + BackPain + Anxiety + Allergic_rhinitis + Reflux_esophagitis + Asthma + Children + Full_meals_eaten + vitD_supp + Initial_days + TotalCharge + Additional_charges, data = medical_encoded, family = binomial)

summary(logres_step3)
display_high_p_value_variables(logres_step3)

# step 4 - remove Hyperlipidemia variable and rerun model
logres_step4 <- glm(formula = ReAdmis ~ Age + GenderFemale + GenderMale + VitD_levels + HighBlood + Stroke + Overweight + Arthritis + Diabetes + BackPain + Anxiety + Allergic_rhinitis + Reflux_esophagitis + Asthma + Children + Full_meals_eaten + vitD_supp + Initial_days + TotalCharge + Additional_charges, data = medical_encoded, family = binomial)

summary(logres_step4)
display_high_p_value_variables(logres_step4)

# step 5 - remove HighBlood variable and rerun model
logres_step5 <- glm(formula = ReAdmis ~ Age + GenderFemale + GenderMale + VitD_levels + Stroke + Overweight + Arthritis + Diabetes + BackPain + Anxiety + Allergic_rhinitis + Reflux_esophagitis + Asthma + Children + Full_meals_eaten + vitD_supp + Initial_days + TotalCharge + Additional_charges, data = medical_encoded, family = binomial)

summary(logres_step5)
display_high_p_value_variables(logres_step5)

# step 6 - remove BackPain variable and rerun model
logres_step6 <- glm(formula = ReAdmis ~ Age + GenderFemale + GenderMale + VitD_levels + Stroke + Overweight + Arthritis + Diabetes + Anxiety + Allergic_rhinitis + Reflux_esophagitis + Asthma + Children + Full_meals_eaten + vitD_supp + Initial_days + TotalCharge + Additional_charges, data = medical_encoded, family = binomial)

summary(logres_step6)
display_high_p_value_variables(logres_step6)

# step 7 - remove Full_meals_eaten variable and rerun model
logres_step7 <- glm(formula = ReAdmis ~ Age + GenderFemale + GenderMale + VitD_levels + Stroke + Overweight + Arthritis + Diabetes + Anxiety + Allergic_rhinitis + Reflux_esophagitis + Asthma + Children + vitD_supp + Initial_days + TotalCharge + Additional_charges, data = medical_encoded, family = binomial)

summary(logres_step7)
display_high_p_value_variables(logres_step7)

# step 8 - remove GenderMale variable and rerun model
logres_step8 <- glm(formula = ReAdmis ~ Age + GenderFemale + VitD_levels + Stroke + Overweight + Arthritis + Diabetes + Anxiety + Allergic_rhinitis + Reflux_esophagitis + Asthma + Children + vitD_supp + Initial_days + TotalCharge + Additional_charges, data = medical_encoded, family = binomial)

summary(logres_step8)
display_high_p_value_variables(logres_step8)

# step 9 - remove GenderFemale variable and rerun model
logres_step9 <- glm(formula = ReAdmis ~ Age + VitD_levels + Stroke + Overweight + Arthritis + Diabetes + Anxiety + Allergic_rhinitis + Reflux_esophagitis + Asthma + Children + vitD_supp + Initial_days + TotalCharge + Additional_charges, data = medical_encoded, family = binomial)

summary(logres_step9)
display_high_p_value_variables(logres_step9)

# step 10 - remove vitD_supp variable and rerun model
logres_step10 <- glm(formula = ReAdmis ~ Age + VitD_levels + Stroke + Overweight + Arthritis + Diabetes + Anxiety + Allergic_rhinitis + Reflux_esophagitis + Asthma + Children + Initial_days + TotalCharge + Additional_charges, data = medical_encoded, family = binomial)

summary(logres_step10)
display_high_p_value_variables(logres_step10)

# step 11 - remove VitD_levels variable and rerun model
logres_step11 <- glm(formula = ReAdmis ~ Age + Stroke + Overweight + Arthritis + Diabetes + Anxiety + Allergic_rhinitis + Reflux_esophagitis + Asthma + Children + Initial_days + TotalCharge + Additional_charges, data = medical_encoded, family = binomial)

summary(logres_step11)
display_high_p_value_variables(logres_step11)

# step 12 - remove Diabetes variable and rerun model
logres_step12 <- glm(formula = ReAdmis ~ Age + Stroke + Overweight + Arthritis + Anxiety + Allergic_rhinitis + Reflux_esophagitis + Asthma + Children + Initial_days + TotalCharge + Additional_charges, data = medical_encoded, family = binomial)

summary(logres_step12)
display_high_p_value_variables(logres_step12)

# step 13 - remove Overweight variable and rerun model
logres_step13 <- glm(formula = ReAdmis ~ Age + Stroke + Arthritis + Anxiety + Allergic_rhinitis + Reflux_esophagitis + Asthma + Children + Initial_days + TotalCharge + Additional_charges, data = medical_encoded, family = binomial)

summary(logres_step13)
display_high_p_value_variables(logres_step13)

# step 14 - remove Children variable
logres_final <- glm(formula = ReAdmis ~ Age + Stroke + Arthritis + Anxiety + Allergic_rhinitis + Reflux_esophagitis + Asthma + Initial_days + TotalCharge + Additional_charges, data = medical_encoded, family = binomial)

summary(logres_final)
display_high_p_value_variables(logres_final)


# Use model to predict probability of readmission
predicted <- as.numeric (predict(logres_final, medical_encoded, type="response"))
predicted <-ifelse(predicted > 0.5,1,0)
predicted <- as.factor(predicted)
str(predicted)
predicted
unique(predicted)

# Convert values from "Yes" and "No" to 1's and 0's
medical_encoded$ReAdmis <- as.factor(medical_encoded$ReAdmis)
medical_encoded$ReAdmis
unique(medical_encoded$ReAdmis)

# Create confusion matrix
confusionMatrix(medical_encoded$ReAdmis,predicted)