# =============================================================================
#  Customer Churn Prediction — End-to-End tidymodels Pipeline
#  Author : Felipe Ramires Terrazas
#  Date   : 2026-03-30
#  Metric : ROC-AUC (binary classification)
# =============================================================================
#
#  Pipeline overview
#  ─────────────────
#  1. Setup & data loading
#  2. Exploratory data prep   (coerce types, inspect missingness)
#  3. Train / validation split
#  4. Preprocessing recipe
#  5. Model specification     (XGBoost — tunable)
#  6. Workflow assembly
#  7. 5-fold cross-validation baseline
#  8. Hyperparameter tuning   (grid search)
#  9. Select best model & final fit
# 10. Predict on test set & write submission
# =============================================================================


# ── 1. SETUP ─────────────────────────────────────────────────────────────────

library(tidyverse)    # readr, dplyr, ggplot2, tidyr, purrr, stringr, forcats
library(tidymodels)   # rsample, recipes, parsnip, workflows, tune, yardstick
library(xgboost)      # XGBoost engine
library(vip)          # Variable importance plots
library(finetune)     # Optional: racing methods for faster tuning

tidymodels_prefer()   # Suppress conflicts with base R (e.g. yardstick::rmse)
set.seed(42)          # Global reproducibility seed


# ── 2. DATA LOADING & TYPE COERCION ──────────────────────────────────────────

# Paths (adjust if running from a different working directory)
KAGGLE_DIR        <- here::here("kaggle")   # or use relative: "kaggle/"
TRAIN_PATH        <- file.path(KAGGLE_DIR, "train.csv")
TEST_PATH         <- file.path(KAGGLE_DIR, "test.csv")
SUBMISSION_PATH   <- file.path(KAGGLE_DIR, "submission.csv")

# --- Load raw data -----------------------------------------------------------
raw_train <- read_csv(TRAIN_PATH, show_col_types = FALSE)
raw_test  <- read_csv(TEST_PATH,  show_col_types = FALSE)

# --- Helper: shared cleaning applied to both train and test ------------------
clean_telco <- function(df) {
  df |>
    mutate(
      # TotalCharges is sometimes imported as character due to blank strings
      # representing customers with tenure == 0 (brand-new subscribers).
      TotalCharges  = parse_number(as.character(TotalCharges)),

      # SeniorCitizen is stored as 0/1 integer but is semantically categorical
      SeniorCitizen = factor(SeniorCitizen, levels = c(0, 1),
                             labels = c("No", "Yes"))
    )
}

train_raw <- raw_train |>
  clean_telco() |>
  mutate(
    # Target: must be a factor; positive class = "Yes" (churned)
    Churn = factor(Churn, levels = c("Yes", "No"))
  )

test_clean <- raw_test |>
  clean_telco()

# Quick data snapshot
glimpse(train_raw)
cat("\nClass balance:\n")
train_raw |> count(Churn) |> mutate(pct = n / sum(n) * 100)

cat("\nMissing values per column (train):\n")
train_raw |>
  summarise(across(everything(), ~ sum(is.na(.)))) |>
  pivot_longer(everything(), names_to = "column", values_to = "n_missing") |>
  filter(n_missing > 0)


# ── 3. TRAIN / VALIDATION SPLIT ──────────────────────────────────────────────
#
#  We split once for a final hold-out check; CV handles model selection.
#  strata = Churn preserves the class imbalance in both splits.

data_split  <- initial_split(train_raw, prop = 0.80, strata = Churn)
df_train    <- training(data_split)
df_val      <- testing(data_split)

cat(sprintf("\nTraining rows: %d  |  Validation rows: %d\n",
            nrow(df_train), nrow(df_val)))


# ── 4. PREPROCESSING RECIPE ──────────────────────────────────────────────────
#
#  Recipe steps are applied in order during prep/bake.
#
#  Column roles:
#    - id          → "ID" role (excluded from modelling)
#    - Churn       → outcome
#    - all others  → predictors

churn_recipe <- recipe(Churn ~ ., data = df_train) |>

  # Remove the id column — it's an identifier, not a signal
  update_role(id, new_role = "ID") |>

  # --- Missing value imputation ---------------------------------------------
  # Numeric: median imputation (robust to skew; suitable for TotalCharges NAs)
  step_impute_median(all_numeric_predictors()) |>

  # Nominal: mode imputation for any rare missing categorical values
  step_impute_mode(all_nominal_predictors()) |>

  # --- Feature engineering --------------------------------------------------
  # Interaction: customers with no internet service have many "No" sub-features;
  # flag this explicitly to help the model distinguish "No service" vs "No add-on"
  step_mutate(
    no_internet = if_else(InternetService == "No", 1L, 0L)
  ) |>

  # --- Encoding -------------------------------------------------------------
  # One-hot encode all nominal predictors (drop first level to avoid collinearity)
  step_dummy(all_nominal_predictors(), one_hot = FALSE) |>

  # --- Scaling --------------------------------------------------------------
  # Center and scale all numeric predictors (required for distance-based
  # methods; harmless for tree models but keeps the pipeline portable)
  step_normalize(all_numeric_predictors()) |>

  # --- Near-zero-variance filter -------------------------------------------
  # Remove columns with near zero variance (can occur after dummy encoding)
  step_nzv(all_predictors())

# Inspect the prepared recipe on the training split
prep(churn_recipe, training = df_train) |> bake(new_data = NULL) |> glimpse()


# ── 5. MODEL SPECIFICATION ───────────────────────────────────────────────────
#
#  XGBoost provides strong out-of-the-box performance on tabular data.
#  Key hyperparameters are marked with tune() for grid search in step 8.

xgb_spec <- boost_tree(
  trees          = tune(),   # number of boosting rounds
  tree_depth     = tune(),   # max depth per tree
  min_n          = tune(),   # minimum observations per leaf
  loss_reduction = tune(),   # gamma — min loss reduction to split
  sample_size    = tune(),   # row subsampling fraction
  mtry           = tune(),   # column subsampling fraction
  learn_rate     = tune()    # eta — step shrinkage
) |>
  set_engine("xgboost", nthread = parallel::detectCores() - 1) |>
  set_mode("classification")


# ── 6. WORKFLOW ──────────────────────────────────────────────────────────────

churn_workflow <- workflow() |>
  add_recipe(churn_recipe) |>
  add_model(xgb_spec)


# ── 7. RESAMPLING: 5-FOLD CROSS-VALIDATION ───────────────────────────────────
#
#  Stratified on the target to maintain class proportions in every fold.

cv_folds <- vfold_cv(df_train, v = 5, strata = Churn)

# Metrics to collect on every fold
eval_metrics <- metric_set(roc_auc, accuracy, f_meas, pr_auc)


# ── 8. HYPERPARAMETER TUNING ─────────────────────────────────────────────────
#
#  Latin hypercube design gives better coverage than a regular grid in
#  high-dimensional hyperparameter spaces for the same number of iterations.

xgb_grid <- grid_latin_hypercube(
  trees(range = c(200L, 1000L)),
  tree_depth(range = c(3L, 8L)),
  min_n(range = c(5L, 30L)),
  loss_reduction(range = c(0, 5)),
  sample_size = sample_prop(range = c(0.6, 1.0)),
  finalize(mtry(), df_train),   # finalize() resolves the # predictors at runtime
  learn_rate(range = c(-3, -1), trans = log10_trans()),
  size = 30                     # 30 candidate hyperparameter combinations
)

cat("\n--- Starting hyperparameter tuning (5-fold CV × 30 grid points) ---\n")
cat("    This may take a few minutes depending on your hardware.\n\n")

tuning_results <- tune_grid(
  churn_workflow,
  resamples = cv_folds,
  grid      = xgb_grid,
  metrics   = eval_metrics,
  control   = control_grid(
    save_pred     = TRUE,
    verbose       = TRUE,
    save_workflow = FALSE
  )
)

# Summarise tuning results
cat("\n--- Top 10 configurations by ROC-AUC ---\n")
show_best(tuning_results, metric = "roc_auc", n = 10) |> print()

# Plot tuning results
autoplot(tuning_results, metric = "roc_auc") +
  labs(title = "XGBoost Hyperparameter Tuning — ROC-AUC",
       subtitle = "5-fold cross-validation across 30 Latin hypercube candidates") +
  theme_minimal()


# ── 9. SELECT BEST MODEL & FINAL FIT ─────────────────────────────────────────

best_params <- select_best(tuning_results, metric = "roc_auc")
cat("\n--- Best hyperparameters ---\n")
print(best_params)

# Finalise the workflow with the best parameters
final_workflow <- finalize_workflow(churn_workflow, best_params)

# last_fit() trains on df_train and evaluates on df_val in one step
final_fit <- last_fit(final_workflow, split = data_split, metrics = eval_metrics)

cat("\n--- Hold-out validation performance ---\n")
collect_metrics(final_fit) |> print()

# Confusion matrix on the hold-out set
collect_predictions(final_fit) |>
  conf_mat(truth = Churn, estimate = .pred_class) |>
  autoplot(type = "heatmap") +
  scale_fill_gradient(low = "#0f1a2a", high = "#2563eb") +
  labs(title = "Confusion Matrix — Hold-out Validation Set") +
  theme_minimal()

# ROC curve on the hold-out set
collect_predictions(final_fit) |>
  roc_curve(truth = Churn, .pred_Yes) |>
  autoplot() +
  labs(title = "ROC Curve — Hold-out Validation Set") +
  theme_minimal()

# Variable importance (top 20 features)
final_fit |>
  extract_fit_parsnip() |>
  vip(num_features = 20, aesthetics = list(fill = "#2563eb", alpha = 0.85)) +
  labs(title = "Feature Importance — Final XGBoost Model") +
  theme_minimal()


# ── 10. PREDICT ON TEST SET & WRITE SUBMISSION ───────────────────────────────
#
#  Extract the fitted workflow trained on the full training data.
#  The sample_submission uses numeric Churn (probability of churn = "Yes").

fitted_workflow <- extract_workflow(final_fit)

# Generate class probabilities on the unseen test set
test_predictions <- predict(fitted_workflow,
                            new_data = test_clean,
                            type = "prob")

# Assemble submission in the exact format: id | Churn (P(Churn = "Yes"))
submission <- test_clean |>
  select(id) |>
  bind_cols(test_predictions) |>
  rename(Churn = .pred_Yes) |>   # probability of churning (positive class)
  select(id, Churn)

# Sanity checks before writing
stopifnot(
  nrow(submission) == nrow(test_clean),
  all(between(submission$Churn, 0, 1)),
  !anyNA(submission)
)

cat(sprintf("\n--- Submission ready: %d rows ---\n", nrow(submission)))
head(submission) |> print()

write_csv(submission, SUBMISSION_PATH)
cat(sprintf("Submission written to: %s\n", SUBMISSION_PATH))
