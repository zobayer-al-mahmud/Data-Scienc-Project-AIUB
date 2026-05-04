# ============================================================
#      INTRODUCTION TO DATA SCIENCE — MIDTERM PROJECT
#             Heart Disease Dataset Analysis
# ============================================================
# Dataset Source:
# https://raw.githubusercontent.com/FarhanMahrab/Data-Science/
# refs/heads/main/MID%20Project/heart_disease.csv
# ============================================================


# ── INSTALL & LOAD PACKAGES ──────────────────────────────────
packages <- c("ggplot2", "dplyr", "caret", "ROSE", "e1071", "randomForest")

for (pkg in packages) {
  if (!pkg %in% rownames(installed.packages())) install.packages(pkg)
  library(pkg, character.only = TRUE)
}

# Ensure dplyr functions are not masked by plyr
if ("package:plyr" %in% search()) detach("package:plyr", unload = TRUE)


# ============================================================
#              A. DATA UNDERSTANDING
# ============================================================

# ── 1. Load Dataset ──────────────────────────────────────────
url  <- "https://raw.githubusercontent.com/FarhanMahrab/Data-Science/refs/heads/main/MID%20Project/heart_disease.csv"
data <- read.csv(url)          # FIX: removed stray closing parenthesis


# ── 2. First Look ────────────────────────────────────────────
head(data)

cat("Rows:", nrow(data), "| Columns:", ncol(data), "\n")

str(data)


# ── 3. Descriptive Statistics ────────────────────────────────
summary(data)

num_cols <- names(data)[sapply(data, is.numeric)]

desc_stats <- data.frame(
  Mean   = sapply(data[, num_cols], function(x) round(mean(x,   na.rm = TRUE), 2)),
  Median = sapply(data[, num_cols], function(x) round(median(x, na.rm = TRUE), 2)),
  SD     = sapply(data[, num_cols], function(x) round(sd(x,     na.rm = TRUE), 2)),
  Min    = sapply(data[, num_cols], function(x) round(min(x,    na.rm = TRUE), 2)),
  Max    = sapply(data[, num_cols], function(x) round(max(x,    na.rm = TRUE), 2))
)
print(desc_stats)


# ── 4. Identify Column Types ─────────────────────────────────
cat("\nNumeric columns:\n");     print(names(data)[sapply(data, is.numeric)])
cat("\nCategorical columns:\n"); print(names(data)[sapply(data, is.character)])


# ============================================================
#        B. INTRODUCE DATA ISSUES (for demonstration)
# ============================================================

set.seed(42)

# Missing values
data$Age[sample(1:nrow(data), 30)]                 <- NA
data$Cholesterol.Level[sample(1:nrow(data), 25)]   <- NA
data$Fasting.Blood.Sugar[sample(1:nrow(data), 20)] <- NA

# Invalid values (negative Age)
data$Age[sample(which(!is.na(data$Age)), 10)] <- -99

# Duplicate rows
data <- rbind(data, data[sample(1:nrow(data), 50), ])

cat("Data issues introduced.\n")
cat("Total rows (with duplicates):", nrow(data), "\n")


# ============================================================
#              C. DATA CLEANING & PREPROCESSING
# ============================================================


# ── 1. Remove Duplicate Rows ─────────────────────────────────
cat("Duplicate rows found:", sum(duplicated(data)), "\n")

data <- data[!duplicated(data), ]

cat("Rows after removing duplicates:", nrow(data), "\n")


# ── 2. Detect & Fix Invalid Data ─────────────────────────────
cat("Invalid Age values (< 0):", sum(data$Age < 0, na.rm = TRUE), "\n")

data$Age[!is.na(data$Age) & data$Age < 0] <- NA

cat("Invalid Age values after fix:", sum(data$Age < 0, na.rm = TRUE), "\n")


# ── 3. Handle Missing Values ─────────────────────────────────
missing_counts <- colSums(is.na(data))
cat("\nMissing values per column:\n")
print(missing_counts[missing_counts > 0])
cat("Total missing values:", sum(is.na(data)), "\n")

# -- Visualize Missing Values
missing_df <- data.frame(
  Column  = names(missing_counts),
  Missing = as.integer(missing_counts)
)
missing_df <- missing_df[missing_df$Missing > 0, ]

ggplot(missing_df, aes(x = reorder(Column, -Missing), y = Missing)) +
  geom_bar(stat = "identity", fill = "tomato", color = "black", width = 0.6) +
  geom_text(aes(label = Missing), vjust = -0.4, size = 3.5) +
  labs(title = "Missing Values per Column",
       x     = "Column",
       y     = "Count of Missing Values") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

# -- Impute: Mean for numeric, Mode for categorical
get_mode <- function(v) {
  u <- unique(v[!is.na(v)])
  u[which.max(tabulate(match(v, u)))]
}

for (col in names(data)) {
  if (is.numeric(data[[col]])) {
    data[[col]][is.na(data[[col]])] <- mean(data[[col]], na.rm = TRUE)
  } else {
    data[[col]][is.na(data[[col]])] <- get_mode(data[[col]])
  }
}

cat("Missing values after imputation:", sum(is.na(data)), "\n")


# ── 4. Data Filtering ────────────────────────────────────────
data <- data %>% dplyr::filter(Age >= 18 & Age <= 80)
cat("Rows after Age filter (18-80):", nrow(data), "\n")

data <- data %>% dplyr::filter(Cholesterol.Level >= 150 & Cholesterol.Level <= 300)
cat("Rows after Cholesterol filter:", nrow(data), "\n")


# ── 5. Detect & Remove Outliers (IQR Method) ─────────────────
num_cols <- names(data)[sapply(data, is.numeric)]

cat("\nOutlier counts per numeric column:\n")
for (col in num_cols) {
  Q1      <- quantile(data[[col]], 0.25)
  Q3      <- quantile(data[[col]], 0.75)
  IQR_val <- IQR(data[[col]])
  n_out   <- sum(data[[col]] < (Q1 - 1.5 * IQR_val) |
                   data[[col]] > (Q3 + 1.5 * IQR_val))
  cat(sprintf("  %-25s: %d outliers\n", col, n_out))
}

# Boxplot: Age (before removal)
ggplot(data, aes(y = Age)) +
  geom_boxplot(fill = "orange", color = "black", outlier.color = "red") +
  labs(title = "Boxplot of Age (Before Outlier Removal)", y = "Age") +
  theme_minimal()

# Remove outliers
for (col in num_cols) {
  Q1      <- quantile(data[[col]], 0.25)
  Q3      <- quantile(data[[col]], 0.75)
  IQR_val <- IQR(data[[col]])
  data    <- data[data[[col]] >= (Q1 - 1.5 * IQR_val) &
                    data[[col]] <= (Q3 + 1.5 * IQR_val), ]
}

cat("Rows after outlier removal:", nrow(data), "\n")


# ── 6. Convert Attributes ────────────────────────────────────

# Numeric -> Categorical: Create Age groups
data$Age.Group <- cut(data$Age,
                      breaks = c(0, 30, 50, 70, 100),
                      labels = c("Young", "Middle-Aged", "Senior", "Elderly"),
                      right  = FALSE)
cat("\nAge Group distribution:\n")
print(table(data$Age.Group))

# Categorical -> Numeric: Label Encoding
cat_cols <- names(data)[sapply(data, is.character)]

for (col in cat_cols) {
  data[[col]] <- as.numeric(as.factor(data[[col]]))
}

data$Age.Group <- as.numeric(data$Age.Group)

cat("\nAll columns after encoding:\n")
str(data)


# ── 7. Normalization (Min-Max) ───────────────────────────────
target_col   <- "Heart.Disease.Status"
num_cols     <- names(data)[sapply(data, is.numeric)]
feature_cols <- setdiff(num_cols, target_col)

normalize <- function(x) (x - min(x)) / (max(x) - min(x))

data[, feature_cols] <- lapply(data[, feature_cols], normalize)

cat("\nSample after Min-Max normalization:\n")
head(data[, feature_cols[1:4]])


# ── 8. Save full normalized data for EDA ─────────────────────
# Keep a copy BEFORE feature selection so EDA has all columns
data_full_normalized <- data


# ── 9. Feature Selection (Top 5 correlated with target) ──────

# FIX: Convert target to numeric first (once, outside the loop)
data[[target_col]] <- as.numeric(as.factor(data[[target_col]]))

# FIX: Compute absolute correlation for each feature and return it properly
cor_vals <- sapply(feature_cols, function(col) {
  abs(cor(data[[col]], data[[target_col]], use = "complete.obs"))
})

top5 <- names(sort(cor_vals, decreasing = TRUE))[1:5]
cat("\nTop 5 features by correlation with target:\n")
print(top5)

data_final <- data[, c(top5, target_col)]
data_final[[target_col]] <- as.factor(data_final[[target_col]])


# ============================================================
#    D. TRAIN / TEST SPLIT  (80 / 20)  -- AFTER CLEANING
# ============================================================

set.seed(42)
train_index <- createDataPartition(data_final[[target_col]], p = 0.8, list = FALSE)

train_data <- data_final[ train_index, ]
test_data  <- data_final[-train_index, ]

cat("\n-- Train / Test Split --\n")
cat("Total rows:", nrow(data_final), "\n")
cat("Train rows:", nrow(train_data),
    sprintf("(%.1f%%)\n", nrow(train_data) / nrow(data_final) * 100))
cat("Test  rows:", nrow(test_data),
    sprintf("(%.1f%%)\n", nrow(test_data)  / nrow(data_final) * 100))

cat("\nTarget distribution in TRAIN:\n")
print(table(train_data[[target_col]]))

cat("\nTarget distribution in TEST:\n")
print(table(test_data[[target_col]]))


# ============================================================
#     E. HANDLE IMBALANCED DATASET  (on train set only)
# ============================================================

cat("\nClass distribution BEFORE balancing (train):\n")
print(table(train_data[[target_col]]))

set.seed(42)
train_balanced <- ROSE(Heart.Disease.Status ~ ., data = train_data, seed = 42)$data

cat("\nClass distribution AFTER balancing (train):\n")
print(table(train_balanced[[target_col]]))


# ============================================================
#         F. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# Use full normalized data (all columns available) for EDA
# Split full normalized data using the same train index
data_full_normalized[[target_col]] <- as.factor(data_full_normalized[[target_col]])

train_full <- data_full_normalized[ train_index, ]

set.seed(42)
df <- ROSE(Heart.Disease.Status ~ ., data = train_full, seed = 42)$data
df$Heart.Disease.Status <- as.numeric(as.character(df$Heart.Disease.Status))


# ── 1. Descriptive Stats by Target Class ─────────────────────
cat("\nDescriptive stats by Heart Disease Status:\n")
df %>%
  dplyr::group_by(Heart.Disease.Status) %>%
  dplyr::summarise(
    Mean_Age         = round(mean(Age), 3),
    Mean_BloodPr     = round(mean(Blood.Pressure), 3),
    Mean_Cholesterol = round(mean(Cholesterol.Level), 3),
    Mean_BMI         = round(mean(BMI), 3)
  ) %>%
  print()


# ── 2. Compare Average of Variable Between Two Groups ────────
group_means <- df %>%
  dplyr::group_by(Heart.Disease.Status) %>%
  dplyr::summarise(Avg_Cholesterol = round(mean(Cholesterol.Level), 3))

print(group_means)

ggplot(group_means, aes(x = factor(Heart.Disease.Status),
                        y = Avg_Cholesterol,
                        fill = factor(Heart.Disease.Status))) +
  geom_bar(stat = "identity", color = "black", width = 0.5) +
  scale_fill_manual(values = c("0" = "steelblue", "1" = "tomato"),
                    labels  = c("No Disease", "Has Disease"),
                    name    = "Heart Disease") +
  labs(title = "Average Cholesterol by Heart Disease Status",
       x     = "Heart Disease Status",
       y     = "Average Cholesterol (Normalized)") +
  theme_minimal()


# ── 3. Compare Spread Across Categories ──────────────────────
ggplot(df, aes(x = factor(Smoking), y = BMI, fill = factor(Smoking))) +
  geom_boxplot(outlier.color = "red", show.legend = FALSE) +
  labs(title = "BMI Spread by Smoking Status",
       x     = "Smoking (1 = No, 2 = Yes)",
       y     = "BMI (Normalized)") +
  theme_minimal()


# ── 4. Heart Disease Class Distribution ──────────────────────
ggplot(df, aes(x = factor(Heart.Disease.Status),
               fill = factor(Heart.Disease.Status))) +
  geom_bar(color = "black", width = 0.5) +
  scale_fill_manual(values = c("0" = "steelblue", "1" = "tomato"),
                    labels  = c("No Disease", "Has Disease"),
                    name    = "Heart Disease") +
  labs(title = "Heart Disease Class Distribution (Balanced Train Set)",
       x     = "Heart Disease Status",
       y     = "Count") +
  theme_minimal()


# ============================================================
#         G. RANDOM FOREST MODEL TRAINING & EVALUATION
# ============================================================

# Ensure target is a factor with valid R level names
train_balanced$Heart.Disease.Status <- factor(
  train_balanced$Heart.Disease.Status,
  levels = c(1, 2),
  labels = c("No_Disease", "Has_Disease")
)

test_data$Heart.Disease.Status <- factor(
  test_data$Heart.Disease.Status,
  levels = c(1, 2),
  labels = c("No_Disease", "Has_Disease")
)

# ── 1. Train Random Forest ───────────────────────────────────
set.seed(42)
rf_model <- randomForest(
  Heart.Disease.Status ~ .,
  data       = train_balanced,
  ntree      = 500,
  mtry       = floor(sqrt(ncol(train_balanced) - 1)),
  importance = TRUE
)

cat("\n-- Random Forest Model Summary --\n")
print(rf_model)


# ── 2. Predictions on Test Set ───────────────────────────────
rf_predictions <- predict(rf_model, newdata = test_data)

cat("\nPrediction distribution on test set:\n")
print(table(rf_predictions))


# ── 3. Confusion Matrix & Metrics ────────────────────────────
cat("\n-- Confusion Matrix --\n")
conf_matrix <- confusionMatrix(rf_predictions, test_data$Heart.Disease.Status)
print(conf_matrix)

cat(sprintf("\nAccuracy   : %.4f\n", conf_matrix$overall["Accuracy"]))
cat(sprintf("Kappa      : %.4f\n", conf_matrix$overall["Kappa"]))
cat(sprintf("Sensitivity: %.4f\n", conf_matrix$byClass["Sensitivity"]))
cat(sprintf("Specificity: %.4f\n", conf_matrix$byClass["Specificity"]))


# ── 4. Variable Importance Plot ───────────────────────────────
varImpPlot(
  rf_model,
  main = "Random Forest - Variable Importance",
  col  = "steelblue",
  pch  = 16
)

# ggplot-style importance chart
importance_df         <- as.data.frame(importance(rf_model))
importance_df$Feature <- rownames(importance_df)

ggplot(importance_df, aes(x = reorder(Feature, MeanDecreaseGini),
                           y = MeanDecreaseGini)) +
  geom_bar(stat = "identity", fill = "steelblue", color = "black", width = 0.6) +
  coord_flip() +
  labs(title = "Random Forest - Feature Importance (Mean Decrease Gini)",
       x     = "Feature",
       y     = "Mean Decrease Gini") +
  theme_minimal()


# ── 5. OOB Error Rate vs Number of Trees ─────────────────────
oob_df <- data.frame(
  Trees = 1:rf_model$ntree,
  OOB   = rf_model$err.rate[, "OOB"]
)

ggplot(oob_df, aes(x = Trees, y = OOB)) +
  geom_line(color = "tomato", linewidth = 0.8) +
  labs(title = "Random Forest - OOB Error Rate vs Number of Trees",
       x     = "Number of Trees",
       y     = "OOB Error Rate") +
  theme_minimal()


# ============================================================
#                     END OF PROJECT
# ============================================================
