# Install.packages
library(glmnet)

# Import CSV to a dataframe
churn_imported <- read.csv("churn_clean.csv")

# View the structure of the data
str(churn_imported)

# Prepare data
# Detect duplicates
sum(duplicated(churn_imported))

# Detect missing values
colSums(is.na(churn_imported))

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
result <- quantitative_outliers(churn_imported, "Outage_sec_perweek")
result <- quantitative_outliers(churn_imported, "Email")
result <- quantitative_outliers(churn_imported, "Contacts")
result <- quantitative_outliers(churn_imported, "Yearly_equip_failure")
result <- quantitative_outliers(churn_imported, "Tenure")
result <- quantitative_outliers(churn_imported, "MonthlyCharge")
result <- quantitative_outliers(churn_imported, "Bandwidth_GB_Year")


# Save prepared dataset to CSV
# Get the current working directory
current_dir <- getwd()

# Specify the CSV file name in the current directory
file_path <- file.path(current_dir, "churn_prepared.csv")

# Save the dataframe to a CSV file in the current directory
write.csv(churn_imported, file = file_path, row.names = FALSE)

# Print a message to confirm the save operation
cat("Dataframe saved to", file_path, "\n")

#Split dataset into training and test sets
# Create dataframe with columns to be used for analysis
churn_reduced = data.frame(churn_imported$Outage_sec_perweek,churn_imported$Email,churn_imported$Contacts,churn_imported$Yearly_equip_failure,churn_imported$MonthlyCharge,churn_imported$Bandwidth_GB_Year)

#make this example reproducible
set.seed(1)

#use 75% of dataset as training set and 25% as test set
# (Zach, 2022)
sample <- sample(c(TRUE, FALSE), nrow(churn_reduced), replace=TRUE, prob=c(0.75,0.25))
churn_train  <- churn_reduced[sample, ]
churn_test   <- churn_reduced[!sample, ]

# Save the split files as CSV
# Specify the CSV file name in the current directory
file_path <- file.path(current_dir, "churn_train.csv")
# Save the dataframe to a CSV file in the current directory
write.csv(churn_train, file = file_path, row.names = FALSE)

# Specify the CSV file name in the current directory
file_path <- file.path(current_dir, "churn_test.csv")
# Save the dataframe to a CSV file in the current directory
write.csv(churn_test, file = file_path, row.names = FALSE)


#Lasso Regression
#define response variable
y <- churn_imported$Tenure

#define matrix of predictor variables
x <- data.matrix(churn_imported[, c('Outage_sec_perweek', 'Email', 'Contacts', 'Yearly_equip_failure', 'MonthlyCharge', 'Bandwidth_GB_Year')])

# Fit the Lasso regression model
#perform k-fold cross-validation to find optimal lambda value
cv_model <- cv.glmnet(x, y, alpha = 1)

#find optimal lambda value that minimizes test MSE
best_lambda <- cv_model$lambda.min
best_lambda

#produce plot of test MSE by lambda value
plot(cv_model)

# Analyze final model
# display coefficients of best model
best_model <- glmnet(x, y, alpha = 1, lambda = best_lambda)
coef(best_model)

# fit model to training data
#use fitted best model to make predictions
y_predicted <- predict(best_model, s = best_lambda, newx = x)

#find SST and SSE
sst <- sum((y - mean(y))^2)
sst
sse <- sum((y_predicted - y)^2)
sse

#find R-Squared
rsq <- 1 - sse/sst
rsq

# calculate MSE
MSE <- sum((y - y_predicted)^2)/10000
MSE