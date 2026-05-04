# ============================================================
#  Introduction to Data Science - Final Term Project
#  Title  : COVID-19 Pandemic Impact Analysis and Prediction
#  Source : Wikipedia - COVID-19 pandemic by country
#  URL    : https://en.wikipedia.org/wiki/COVID-19_pandemic_by_country_and_territory
#  Models : Random Forest Regression | K-Means Clustering
# ============================================================


# ============================================================
# SECTION 0 - Load Libraries
# ============================================================
install.packages(c("rvest","dplyr","ggplot2","randomForest",
                   "caret","cluster","factoextra","corrplot","scales"))


library(rvest)
library(dplyr)
library(stringr)
library(ggplot2)
library(scales)
library(randomForest)
library(caret)
library(cluster)
library(factoextra)
library(corrplot)


# ============================================================
# SECTION 1 - RESEARCH OBJECTIVE
# ============================================================

# Objective:
# To analyze COVID-19 data across countries and:
# (1) Predict total deaths using Random Forest Regression
# (2) Classify countries into Low / Medium / High risk groups
#     using K-Means Clustering


# ============================================================
# SECTION 2 - DATA COLLECTION (WEB SCRAPING)
# ============================================================

# FIX: Wikipedia's first wikitable is a map image table (1 col, 3 rows).
# We scan ALL wikitables and select the one with the most rows,
# which is reliably the country statistics table.

url  <- "https://en.wikipedia.org/wiki/COVID-19_pandemic_by_country_and_territory"
page <- read_html(url)

all_tables <- html_nodes(page, "table.wikitable")
cat("Total wikitables found:", length(all_tables), "\n")

# Find the table with the most rows
best_idx  <- 1
max_rows  <- 0
for (i in seq_along(all_tables)) {
  temp <- tryCatch(html_table(all_tables[[i]], fill = TRUE),
                   error = function(e) NULL)
  if (!is.null(temp) && nrow(temp) > max_rows) {
    max_rows <- nrow(temp)
    best_idx <- i
  }
}

df_raw <- html_table(all_tables[[best_idx]], fill = TRUE)
cat("Selected table index:", best_idx,
    "| Rows:", nrow(df_raw),
    "| Cols:", ncol(df_raw), "\n")


# ============================================================
# SECTION 3 - DATA UNDERSTANDING AND EXPLORATION
# ============================================================

cat("\n--- Raw Column Names ---\n")
print(names(df_raw))

cat("\n--- First 6 Rows ---\n")
print(head(df_raw))

cat("\n--- Dataset Dimensions ---\n")
cat("Rows:", nrow(df_raw), "| Columns:", ncol(df_raw), "\n")


# ============================================================
# SECTION 4 - DATA PREPROCESSING
# ============================================================

df <- df_raw

# ------------------------------------------------------------------
# Step 4.1 - Identify columns by pattern matching
# FIX: Wikipedia tables have multi-row merged headers whose text
# gets concatenated by html_table(). We locate each column by
# searching for keywords rather than assuming a fixed column order.
# ------------------------------------------------------------------

col_names_lower <- tolower(names(df))

find_col <- function(patterns) {
  for (pat in patterns) {
    idx <- grep(pat, col_names_lower)
    if (length(idx) > 0) return(idx[1])
  }
  return(NA)
}

idx_country <- find_col(c("country", "location", "territory", "state"))
idx_cases   <- find_col(c("cases", "confirmed", "total case"))
idx_deaths  <- find_col(c("death", "fatal"))
# deaths_per_million: prefer "million" column that also mentions death
idx_dpm     <- {
  cands <- grep("million", col_names_lower)
  death_cands <- intersect(cands, grep("death", col_names_lower))
  if (length(death_cands) > 0) death_cands[1]
  else if (length(cands) > 0)  cands[1]
  else NA
}

cat("\nColumn mapping:\n")
cat("  country           -> col", idx_country, ":", names(df)[idx_country], "\n")
cat("  cases             -> col", idx_cases,   ":", names(df)[idx_cases],   "\n")
cat("  deaths            -> col", idx_deaths,  ":", names(df)[idx_deaths],  "\n")
cat("  deaths_per_million-> col", idx_dpm,     ":", names(df)[idx_dpm],     "\n")

# Stop with a clear message if any critical column is missing
missing_cols <- c(country = idx_country, cases = idx_cases, deaths = idx_deaths)
if (any(is.na(missing_cols))) {
  stop("Could not locate required columns. Run names(df_raw) and adjust find_col() patterns.\n",
       "Available columns:\n", paste(names(df_raw), collapse = "\n"))
}

# Subset to only the four columns we need
keep <- c(idx_country, idx_cases, idx_deaths,
          if (!is.na(idx_dpm)) idx_dpm else integer(0))
df   <- df[, keep]

# Assign clean names
if (!is.na(idx_dpm)) {
  names(df) <- c("country", "cases", "deaths", "deaths_per_million")
} else {
  names(df) <- c("country", "cases", "deaths")
  df$deaths_per_million <- NA_real_          # placeholder; filled below
  cat("WARNING: deaths_per_million column not found; will be calculated later.\n")
}

# ------------------------------------------------------------------
# Step 4.2 - Drop header/blank rows and aggregate rows
# ------------------------------------------------------------------
header_like <- c("Country", "country", "Location", "Territory",
                 "World", "European Union",
                 "Africa", "Asia", "Europe", "Americas",
                 "Oceania", "North America", "South America")

df <- df[!df$country %in% header_like, ]
df <- df[!is.na(df$country) & df$country != "", ]

# ------------------------------------------------------------------
# Step 4.3 - Clean numeric columns (strip commas, notes, symbols)
# ------------------------------------------------------------------
clean_num <- function(x) {
  x <- gsub(",", "", x)
  x <- gsub("\\[.*?\\]", "", x)   # remove footnote refs like [3]
  x <- gsub("[^0-9.]", "", x)
  as.numeric(x)
}

df$cases              <- clean_num(df$cases)
df$deaths             <- clean_num(df$deaths)
df$deaths_per_million <- clean_num(df$deaths_per_million)

# ------------------------------------------------------------------
# Step 4.4 - Remove rows missing core numeric values
# ------------------------------------------------------------------
df <- df[!is.na(df$cases) & !is.na(df$deaths), ]
df <- df[df$cases > 0 & df$deaths >= 0, ]

cat("\n--- After Cleaning ---\n")
cat("Rows:", nrow(df), "\n")

# ------------------------------------------------------------------
# Step 4.5 - Feature Engineering
# ------------------------------------------------------------------
df$death_rate <- df$deaths / df$cases       # Case fatality rate

# If deaths_per_million was absent, approximate it
# (deaths / cases * 1e6 is a rough stand-in; ideally use population)
if (all(is.na(df$deaths_per_million))) {
  cat("Computing deaths_per_million as deaths / cases * 1e6 (approximation).\n")
  df$deaths_per_million <- df$death_rate * 1e6
}

df$cases_per_1000 <- df$cases / 1000

cat("\n--- Summary Statistics ---\n")
print(summary(df[, c("cases", "deaths", "deaths_per_million", "death_rate")]))

# ------------------------------------------------------------------
# Step 4.6 - Check missing values
# ------------------------------------------------------------------
cat("\n--- Missing Values ---\n")
print(colSums(is.na(df)))

# Drop any remaining incomplete rows (needed for models)
df <- df[complete.cases(df[, c("cases", "deaths", "deaths_per_million", "death_rate")]), ]
cat("Rows after dropping remaining NAs:", nrow(df), "\n")

# ------------------------------------------------------------------
# Step 4.7 - Outlier Detection (IQR method)
# ------------------------------------------------------------------
Q1      <- quantile(df$deaths, 0.25, na.rm = TRUE)
Q3      <- quantile(df$deaths, 0.75, na.rm = TRUE)
IQR_val <- Q3 - Q1
outliers <- df[df$deaths > Q3 + 1.5 * IQR_val, ]
cat("\n--- Outliers in Deaths (IQR method) ---\n")
print(outliers[, c("country", "deaths")])

# ------------------------------------------------------------------
# Step 4.8 - Normalize features for clustering
# ------------------------------------------------------------------
df_norm <- as.data.frame(
  scale(df[, c("cases", "deaths", "deaths_per_million", "death_rate")])
)

# ------------------------------------------------------------------
# Step 4.9 - Save cleaned dataset
# ------------------------------------------------------------------
write.csv(df, "covid19_cleaned.csv", row.names = FALSE)
cat("\nCleaned dataset saved: covid19_cleaned.csv\n")


# ============================================================
# SECTION 5 - EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

if (!dir.exists("plots")) dir.create("plots")

# Plot 1: Top 20 Countries by Deaths per Million
top20 <- head(arrange(df, desc(deaths_per_million)), 20)

p1 <- ggplot(top20, aes(x = reorder(country, deaths_per_million),
                        y = deaths_per_million)) +
  geom_bar(stat = "identity", fill = "#c0392b") +
  coord_flip() +
  scale_y_continuous(labels = comma) +
  labs(title = "Top 20 Countries: Deaths per Million",
       x = "Country", y = "Deaths per Million") +
  theme_minimal(base_size = 12)

ggsave("plots/01_top20_deaths_per_million.png", p1, width = 10, height = 7)
cat("Plot 1 saved.\n")


# Plot 2: Distribution of Total Cases
p2 <- ggplot(df, aes(x = cases)) +
  geom_histogram(bins = 30, fill = "#2980b9", color = "white") +
  scale_x_log10(labels = comma) +
  labs(title = "Distribution of Total Cases (Log Scale)",
       x = "Total Cases", y = "Number of Countries") +
  theme_minimal(base_size = 12)

ggsave("plots/02_distribution_cases.png", p2, width = 9, height = 6)
cat("Plot 2 saved.\n")


# Plot 3: Distribution of Total Deaths
p3 <- ggplot(df, aes(x = deaths)) +
  geom_histogram(bins = 30, fill = "#e67e22", color = "white") +
  scale_x_log10(labels = comma) +
  labs(title = "Distribution of Total Deaths (Log Scale)",
       x = "Total Deaths", y = "Number of Countries") +
  theme_minimal(base_size = 12)

ggsave("plots/03_distribution_deaths.png", p3, width = 9, height = 6)
cat("Plot 3 saved.\n")


# Plot 4: Cases vs Deaths Scatter
p4 <- ggplot(df, aes(x = cases, y = deaths)) +
  geom_point(colour = "#8e44ad", alpha = 0.7, size = 2.5) +
  geom_smooth(method = "lm", se = FALSE, colour = "red") +
  scale_x_log10(labels = comma) +
  scale_y_log10(labels = comma) +
  labs(title = "Cases vs Deaths by Country (Log Scale)",
       x = "Total Cases", y = "Total Deaths") +
  theme_minimal(base_size = 12)

ggsave("plots/04_cases_vs_deaths.png", p4, width = 9, height = 6)
cat("Plot 4 saved.\n")


# Plot 5: Correlation Heatmap
num_cols <- df[, c("cases", "deaths", "deaths_per_million", "death_rate")]
cor_mat  <- cor(num_cols, use = "complete.obs")

png("plots/05_correlation_heatmap.png", width = 700, height = 600, res = 120)
corrplot(cor_mat, method = "color", addCoef.col = "black",
         tl.col = "black", tl.srt = 45,
         col = colorRampPalette(c("#2980b9", "white", "#c0392b"))(200),
         title = "Correlation Heatmap", mar = c(0, 0, 1, 0))
dev.off()
cat("Plot 5 saved.\n")


# ============================================================
# SECTION 6 - MODEL 1: RANDOM FOREST REGRESSION
# Predict: Total Deaths
# Features: cases, deaths_per_million, death_rate
# ============================================================

cat("\n=== MODEL 1: RANDOM FOREST REGRESSION ===\n")

set.seed(42)

# Prepare data
rf_data <- df[, c("cases", "deaths_per_million", "death_rate", "deaths")]
rf_data <- rf_data[complete.cases(rf_data), ]

# Guard: need at least 10 rows for a meaningful split
if (nrow(rf_data) < 10) stop("Too few complete rows for modelling (< 10).")

# Train / Test Split (80 / 20)
train_index <- createDataPartition(rf_data$deaths, p = 0.8, list = FALSE)
train_data  <- rf_data[ train_index, ]
test_data   <- rf_data[-train_index, ]

cat("Training samples:", nrow(train_data), "\n")
cat("Test samples    :", nrow(test_data),  "\n")

# Train Random Forest
rf_model <- randomForest(deaths ~ cases + deaths_per_million + death_rate,
                         data       = train_data,
                         ntree      = 300,
                         importance = TRUE)

cat("\n--- Random Forest Model ---\n")
print(rf_model)

# Predictions on Test Set
rf_preds <- predict(rf_model, newdata = test_data)

# Performance Metrics
mae  <- mean(abs(rf_preds - test_data$deaths))
rmse <- sqrt(mean((rf_preds - test_data$deaths)^2))
r2   <- cor(rf_preds, test_data$deaths)^2

cat("\n--- Regression Performance (Test Set) ---\n")
cat(sprintf("  R²   : %.4f\n", r2))
cat(sprintf("  MAE  : %.2f\n", mae))
cat(sprintf("  RMSE : %.2f\n", rmse))


# Plot 6: Actual vs Predicted
pred_df <- data.frame(Actual = test_data$deaths, Predicted = rf_preds)

p6 <- ggplot(pred_df, aes(x = Actual, y = Predicted)) +
  geom_point(colour = "#27ae60", size = 3, alpha = 0.8) +
  geom_abline(slope = 1, intercept = 0,
              colour = "red", linetype = "dashed", linewidth = 1) +
  scale_x_log10(labels = comma) +
  scale_y_log10(labels = comma) +
  labs(title    = "RF Regression: Actual vs Predicted Deaths",
       subtitle  = paste0("R² = ", round(r2, 4),
                          "  |  RMSE = ", round(rmse, 0)),
       x = "Actual Deaths", y = "Predicted Deaths") +
  theme_minimal(base_size = 12)

ggsave("plots/06_rf_actual_vs_predicted.png", p6, width = 9, height = 7)
cat("Plot 6 saved.\n")


# Plot 7: Variable Importance
imp_df <- data.frame(
  Feature    = rownames(importance(rf_model)),
  Importance = importance(rf_model)[, "%IncMSE"]
)

p7 <- ggplot(imp_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "#e74c3c") +
  coord_flip() +
  labs(title = "Random Forest - Variable Importance",
       x = "Feature", y = "% Increase in MSE") +
  theme_minimal(base_size = 12)

ggsave("plots/07_variable_importance.png", p7, width = 8, height = 5)
cat("Plot 7 saved.\n")


# ============================================================
# SECTION 7 - MODEL 2: K-MEANS CLUSTERING (FIXED VERSION)
# ============================================================

cat("\n=== MODEL 2: K-MEANS CLUSTERING ===\n")

set.seed(42)

# ------------------------------------------------------------
# FIX 1: Use LOG TRANSFORMATION (handles skewness)
# ------------------------------------------------------------
df_cluster <- df

df_cluster$log_cases  <- log(df_cluster$cases + 1)
df_cluster$log_deaths <- log(df_cluster$deaths + 1)
df_cluster$log_dpm    <- log(df_cluster$deaths_per_million + 1)

# ------------------------------------------------------------
# FIX 2: Use meaningful features only
# ------------------------------------------------------------
cluster_features <- df_cluster[, c("log_cases", "log_deaths", "log_dpm")]

# ------------------------------------------------------------
# FIX 3: Scale features
# ------------------------------------------------------------
df_norm <- as.data.frame(scale(cluster_features))

# ------------------------------------------------------------
# FIX 4: Apply K-Means
# ------------------------------------------------------------
k <- 3
kmeans_model <- kmeans(df_norm, centers = k, nstart = 50)

df$cluster <- kmeans_model$cluster

# ------------------------------------------------------------
# FIX 5: Label clusters properly
# ------------------------------------------------------------
cluster_means <- df %>%
  group_by(cluster) %>%
  summarise(mean_dpm = mean(deaths_per_million, na.rm = TRUE)) %>%
  arrange(mean_dpm) %>%
  mutate(label = c("Low Risk", "Medium Risk", "High Risk"))

df$risk_group <- cluster_means$label[match(df$cluster, cluster_means$cluster)]
df$risk_group <- factor(df$risk_group,
                        levels = c("Low Risk", "Medium Risk", "High Risk"))

# ------------------------------------------------------------
# FIX 6: Evaluate clustering
# ------------------------------------------------------------
sil       <- silhouette(kmeans_model$cluster, dist(df_norm))
sil_score <- mean(sil[, 3])

cat("\n--- Cluster Sizes (FIXED) ---\n")
print(table(df$risk_group))

cat(sprintf("\nSilhouette Score: %.4f\n", sil_score))

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
cat("\n--- Cluster Summary ---\n")
print(df %>%
        group_by(risk_group) %>%
        summarise(Countries   = n(),
                  Mean_Deaths = round(mean(deaths, na.rm = TRUE)),
                  Mean_DPM    = round(mean(deaths_per_million, na.rm = TRUE), 1)))

# ------------------------------------------------------------
# Plot: Improved cluster visualization
# ------------------------------------------------------------
p8 <- ggplot(df, aes(x = cases, y = deaths, colour = risk_group)) +
  geom_point(size = 3, alpha = 0.8) +
  scale_x_log10(labels = comma) +
  scale_y_log10(labels = comma) +
  scale_colour_manual(values = c("Low Risk"    = "#27ae60",
                                 "Medium Risk" = "#f39c12",
                                 "High Risk"   = "#c0392b")) +
  labs(title  = "Improved K-Means Clustering",
       x      = "Total Cases (log scale)",
       y      = "Total Deaths (log scale)",
       colour = "Risk Group") +
  theme_minimal(base_size = 12)

ggsave("plots/08_kmeans_clusters_fixed.png", p8, width = 10, height = 7)
cat("Improved cluster plot saved.\n")


# Plot 9: Cluster Boxplot (Deaths per Million)
p9 <- ggplot(df, aes(x = risk_group, y = deaths_per_million, fill = risk_group)) +
  geom_boxplot(alpha = 0.8) +
  scale_fill_manual(values = c("Low Risk"    = "#27ae60",
                               "Medium Risk" = "#f39c12",
                               "High Risk"   = "#c0392b")) +
  labs(title = "Deaths per Million by Risk Group",
       x = "Risk Group", y = "Deaths per Million") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")

ggsave("plots/09_cluster_boxplot.png", p9, width = 8, height = 6)
cat("Plot 9 saved.\n")


# Plot 10: Elbow Plot (optimal k)
wss <- sapply(1:8, function(k) {
  kmeans(df_norm, centers = k, nstart = 25)$tot.withinss
})

elbow_df <- data.frame(k = 1:8, wss = wss)

p10 <- ggplot(elbow_df, aes(x = k, y = wss)) +
  geom_line(colour = "#2980b9", linewidth = 1.2) +
  geom_point(colour = "#c0392b", size = 3) +
  labs(title = "Elbow Plot - Optimal Number of Clusters",
       x = "Number of Clusters (k)", y = "Total Within-Cluster SS") +
  theme_minimal(base_size = 12)

ggsave("plots/10_elbow_plot.png", p10, width = 8, height = 5)
cat("Plot 10 saved.\n")


# ============================================================
# SECTION 8 - RESULTS SUMMARY
# ============================================================

cat("\n============================================================\n")
cat("RESULTS SUMMARY\n")
cat("============================================================\n")

cat("\n--- Random Forest Regression ---\n")
cat(sprintf("  R²   : %.4f\n", r2))
cat(sprintf("  MAE  : %.2f deaths\n", mae))
cat(sprintf("  RMSE : %.2f deaths\n", rmse))

cat("\n--- K-Means Clustering ---\n")
cat(sprintf("  Silhouette Score : %.4f\n", sil_score))
cat("  Risk Groups      : Low Risk | Medium Risk | High Risk\n")
cat("  Countries per group:\n")
print(table(df$risk_group))




