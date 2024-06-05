library(tidyverse)
library(dplyr)
library(datasets)
library(tidyr)
library(arules)

# Read CSV into a dataframe
medt3 <- read.csv("~/MSDA/D212 Data Mining 2/D212 performance assessments/Task3/d8fj4l3d9s94jfu43kf9 (1)/medical_market_basket.csv")
dim(medt3)

# Remove blank rows
cleaned_medt3 <- medt3[!apply(medt3 == "", 1, all),]
dim(cleaned_medt3)

# Save cleaned data as CSV
write.csv(cleaned_medt3, "~/MSDA/D212 Data Mining 2/D212 performance assessments/Task3/PartC1.csv", row.names=FALSE)

# Add Id column
cleaned_medt3$Id <- factor(seq.int(nrow(cleaned_medt3)))

# Factorize dataset
cleaned_medt3 <- as.data.frame(unclass(cleaned_medt3), stringsAsFactors = TRUE)

# Pivot the dataframe
pre_trans <- pivot_longer(cleaned_medt3, cols = 1:20, names_to = "ItemNo", values_to = "Product")

head(pre_trans)

# Keep only the relevant variables
pre_trans <- pre_trans[, c(1,3)]

# Remove empty rows
pre_trans <- pre_trans[!(pre_trans$Product == ""), ]

# Create a list
list_data <- as.data.frame(pre_trans)

# Split the list by Id and Product
list_data <- split(list_data$Product, list_data$Id)

# Check structure of split list
str(list_data)

# Transactionalized dataset
basket <- as(list_data, "transactions")

# Convert basket to matrix
basket <- as(basket, "matrix")

str(basket)
dim(basket)

# Run Apriori function on transactions dataset to uncover association rules
arules <- apriori(basket, control = list(verbose = F), paramet = list(supp = 0.008, conf = 0.4, minlen = 2))

# Remove redundant rules
redundant_r <- is.redundant(arules)
refined_arules <- arules[!redundant_r]

# C3
# Convert the association rules to a data frame
rules_df <- as.data.frame(inspect(refined_arules))

# Save the association rules data frame to a CSV file
write.csv(rules_df, file = "PartC3.csv", row.names = FALSE)

# Inspect top rules sorted by parameter 'lift' in decreasing order
inspect(head(sort(refined_arules, by = "lift", decreasing = T), 10))

summary(refined_arules)