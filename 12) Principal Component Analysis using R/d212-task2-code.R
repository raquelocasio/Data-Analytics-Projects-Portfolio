library(tidyverse)


# Import CSV file to a dataframe
med_import <- read_csv("~/MSDA/D212 Data Mining 2/D212 performance assessments/Task2/d8fj4l3d9s94jfu43kf9/medical_clean.csv")

# Create a new dataframe with only the specified columns
medt2 = data.frame(med_import$Income,med_import$VitD_levels,med_import$Initial_days,med_import$Additional_charges)

# Print the new dataframe
cat("\nNew Dataframe with Selected Columns:\n")
print(medt2)

# Standardize the data (centering and scaling)
scaled_data <- scale(medt2)

# Save scaled data to a CSV file
scaled_data_df <- as.data.frame(scaled_data)
write.csv(scaled_data_df, file = "PartC2_scaled_data.csv", row.names = FALSE)

# Calculate principal components
results <- prcomp(medt2, scale = TRUE)
print(results)

# reverse the signs
results$rotation <- -1*results$rotation

# Part D1, display principal components
results$rotation

# Visualize results
biplot(results, scale = 0)

# D2
# Calculate the correlation matrix
cor_matrix <- cor(medt2)

# Calculate the eigenvalues
eigenvalues <- eigen(cor_matrix)$values

# Plot the scree plot
plot(1:length(eigenvalues), eigenvalues, type = "b", 
     main = "Scree Plot for Kaiser Criterion",
     xlab = "Principal Component", ylab = "Eigenvalue")

# Add a line at eigenvalue = 1
abline(h = 1, col = "red", lty = 2)

# Add labels
text(1:length(eigenvalues), eigenvalues, labels = round(eigenvalues, 2), pos = 3, cex = 0.8)

# Identify the components based on Kaiser criterion (eigenvalue > 1)
num_components <- sum(eigenvalues > 1)
abline(v = num_components + 0.5, col = "blue", lty = 2)
text(num_components + 0.5, max(eigenvalues), paste("Components =", num_components), pos = 3, col = "blue", cex = 0.8)

#calculate total variance explained by each principal component
results$sdev^2 / sum(results$sdev^2)

# D4, total variance captured by principal components
summary(results)