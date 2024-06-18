# Install necessary packages
library(forecast)
library(tseries)
library(ggplot2)
library(seasonal)
library(caTools)
library(astsa)
library(graphics)

# Read data from CSV file into a dataframe
telecodata <- read.csv("teleco_time_series.csv")

# Data preparation
# show dimensions, structure, and sample data
str(telecodata)
# check for NA values
colSums(is.na(telecodata))
# check for null values
is.null(telecodata)
# check for duplicates in Day column
anyDuplicated((telecodata$Day))
# save prepared dataset to CSV
write.csv(telecodata, "telecodata_fulldataset.csv", row.names=FALSE)

# C1
# Create a time series plot
ggplot(telecodata, aes(x = Day, y = Revenue)) +
  geom_line() +
  labs(title = "Time Series Plot of Revenue",
       x = "Day",
       y = "Revenue in Millions")

# C2
# verify dataframe
is.data.frame(telecodata)
head(telecodata)
# convert Revenue column to vector
telecodata_vector <- telecodata[['Revenue']]
# verify vector
is.vector(telecodata_vector)
# convert to time series
telecodata_ts <- ts(telecodata_vector, frequency = 731/24)
# verify time series object
is.ts(telecodata_ts)



# C3
# ADFuller test for stationarity
adf_result <- adf.test(telecodata_ts)

# Print ADFuller test results
print(adf_result)

# Additional information based on ADFuller test
if (adf_result$p.value < 0.05) {
  cat("The time series is likely stationary (p-value < 0.05)\n")
} else {
  cat("The time series is likely non-stationary (p-value >= 0.05)\n")
}

# C5
#make this example reproducible
set.seed(1)

#use 70% of dataset as training set and 30% as test set
sample <- sample.split(telecodata$Day, SplitRatio = 0.7)
train  <- subset(telecodata_vector, sample == TRUE)
write.csv(train, "train_data.csv", row.names=FALSE)
test   <- subset(telecodata_vector, sample == FALSE)
write.csv(train, "test_data.csv", row.names=FALSE)

# convert train set to time series
telecodatatrain_ts <- ts(train, frequency = 731/24)
# verify time series object
is.ts(telecodatatrain_ts)
# convert test set to time series
telecodatatest_ts <- ts(test, frequency = 731/24)
# verify time series object
is.ts(telecodatatest_ts)


# D1
# Decomposed time series
decomposition <- decompose(telecodata_ts)

# plot seasonality
plot(decomposition$seasonal, main = "Seasonal Plot")

# plot trend
plot(decomposition$trend, main = "Trend Plot")

# Calculate and plot autocorrelation
acf_result <- acf(telecodata$Revenue, lag.max = 30) 
plot(acf_result, main = "Autocorrelation Function for Original Time Series")

# plot spectral density
specden <- spectrum(telecodata_ts)
# Plot the spectrum with a title
plot(specden, main = "Spectral Density Plot")

# plot decomposed time series
stl <- stl(telecodata_ts, s.window="period")
autoplot(stl, main = "Decomposed Time Series Plot")

# plot residuals
# Calculate the residuals
residuals <- telecodata$Revenue - (decomposition$trend + decomposition$seasonal + decomposition$random)

# Plot
ggplot(telecodata, aes(x = Day, y = residuals)) +
  geom_line(color = "orange") +
  labs(title = "Residuals of the Time Series",
       x = "Day",
       y = "Residuals") +
  theme_minimal()


# D2
# automate model selection
auto.arima(telecodata_ts, seasonal = T)

# fit model
fit <- Arima(telecodata_ts, order = c(1,1,0))

# run summary statistics on model
summary(fit)

# review residual pattern
checkresiduals(fit)


# D3
# forecast using model
forecast <- forecast(fit, h = 183)

# visualize forecast based on training data
plot(forecast)

# add testing data points for comparison
lines(test)

# visualize p-values for Ljung-Box statistics
Box.test(forecast, lag = 1, type = c("Ljung-Box"))


# E2
# forecast with new model using test data converted to time series
sarima.for(telecodatatest_ts, n.ahead = 731, 1,1,0, plot.all = T, xlab="Months", main = "Revenue Forecast vs. Test Data")

# annotations
text(5,32,labels="Test Data")
arrows(5,29,5,15)
text(15,70,labels="Forecast")
arrows(15,66,15,34)
text(27,10, labels="Confidence Intervals")
arrows(25,12,25,35)
arrows(26,12,26,50)
arrows(27,12,27,69)
arrows(28,12,28,85)