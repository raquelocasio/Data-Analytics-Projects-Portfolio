library(factoextra)
library(cluster)
library(dplyr)
library(readr)


# Import CSV file to a dataframe
med_import <- read_csv("~/MSDA/D212 Data Mining 2/D212 performance assessments/Task1/d8fj4l3d9s94jfu43kf9/medical_clean.csv")

# Value to filter on
filter_value <- 'Yes'

# Select records based on the value in the 'ReAdmis' column
selected_records <- med_import[med_import$ReAdmis == filter_value, ]

# Specify the columns to be copied
columns_to_copy <- c("Income", "VitD_levels", "TotalCharge", "Additional_charges")

# Create a new dataframe with only the specified columns
medt1 <- selected_records[, columns_to_copy]

# Print the new dataframe
cat("\nNew Dataframe with Selected Columns:\n")
print(medt1)

# Detect duplicates
duplicated(medt1)

# Detect missing data
colSums(is.na(medt1))

# Detect outliers
# Function to detect outliers using z-scores
detect_outliers <- function(data, column_names) {
  # Initialize a list to store results
  results <- list()
  
  for (column_name in column_names) {
    # Calculate the z-scores using the scale function
    data[, "z_score"] <- scale(data[, column_name])
    
    # Count the outliers based on the threshold (z-score > 3 or z-score < -3)
    outliers_count <- sum(data$z_score > 3 | data$z_score < -3)
    
    # Store the results in the list
    results[[column_name]] <- list(
      outliers_count = outliers_count,
      outliers = data %>% filter(z_score > 3 | z_score < -3)
    )
  }
  
  return(results)
}

# Specify a list of column names to analyze for outliers
columns_to_check <- c("Income", "VitD_levels", "TotalCharge", "Additional_charges")

# Call the function to detect outliers
outlier_results <- detect_outliers(medt1, columns_to_check)

# Print the results
for (column_name in columns_to_check) {
  cat("Number of outliers in", column_name, ":", outlier_results[[column_name]]$outliers_count, "\n")
}

# Part C4, provide copy of cleaned dataset
write.csv(medt1, "~/MSDA/D212 Data Mining 2/D212 performance assessments/Task1/partC4.csv", row.names=FALSE)

# Part D, perform k-means cluster analysis
#scale each variable to have a mean of 0 and sd of 1
medt1 <- scale(medt1)

#view first six rows of dataset
head(medt1)

# Draw plots to determine optimal number for k
# create plot for number of clusters vs. the total within sum of squares
fviz_nbclust(medt1, kmeans, method = "wss")

#calculate gap statistic based on number of clusters
gap_stat <- clusGap(medt1,
                    FUN = kmeans,
                    nstart = 25,
                    K.max = 10,
                    B = 50)

#plot number of clusters vs. gap statistic
fviz_gap_stat(gap_stat)

# Run k-means cluster analysis
# make this example reproducible
set.seed(1)

# perform k-means clustering with k = 5 clusters
km <- kmeans(medt1, centers = 5, nstart = 25)

# view results
km

# Determine quality of clusters
set.seed(123)  # for reproducibility
k <- 5
kmeans_result <- kmeans(medt1, centers = k)
inertia <- kmeans_result$tot.withinss
inertia