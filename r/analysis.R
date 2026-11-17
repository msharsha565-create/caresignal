#!/usr/bin/env Rscript
# Independent check of the PyTorch readmission model, base R only.
#
#   Rscript r/analysis.R [artifact_dir]     # default: artifacts  (real data: artifacts/uci)
#
# 1. Logistic regression on the same training split, features and median imputation.
# 2. Held-out AUROC: neural net vs. logistic regression.
# 3. Calibration of the neural net by predicted-risk decile.
# Writes <artifact_dir>/r_validation.md

args <- commandArgs(trailingOnly = TRUE)
art <- if (length(args) >= 1) args[1] else "artifacts"

cohort <- read.csv(file.path(art, "cohort.csv"), stringsAsFactors = FALSE)
preds  <- read.csv(file.path(art, "predictions.csv"), stringsAsFactors = FALSE)
df <- merge(cohort, preds[, c("patient_id", "split", "predicted_risk")], by = "patient_id")

df$sex_male <- as.integer(df$sex == "M")
df$discharged_snf <- as.integer(df$discharge_disposition == "snf")
df$discharged_home_health <- as.integer(df$discharge_disposition == "home_health")

candidates <- c("age", "length_of_stay", "prior_admissions_12m", "num_medications",
                "num_diagnoses", "charlson_index", "hba1c", "creatinine", "systolic_bp",
                "bmi", "chf", "copd", "diabetes", "ckd", "lives_alone",
                "sex_male", "discharged_snf", "discharged_home_health")
informative <- function(f) {
  v <- df[[f]]
  sum(!is.na(v)) > 0 && length(unique(v[!is.na(v)])) > 1
}
features <- Filter(informative, intersect(candidates, names(df)))

train <- df$split == "train"
for (f in features) {
  med <- median(df[[f]][train], na.rm = TRUE)   # same training-set imputation as the Python model
  df[[f]][is.na(df[[f]])] <- med
}

fit <- glm(as.formula(paste("readmitted_30d ~", paste(features, collapse = " + "))),
           data = df[train, ], family = binomial())

co <- summary(fit)$coefficients
or <- data.frame(
  term = rownames(co),
  odds_ratio = exp(co[, "Estimate"]),
  ci_low = exp(co[, "Estimate"] - 1.96 * co[, "Std. Error"]),
  ci_high = exp(co[, "Estimate"] + 1.96 * co[, "Std. Error"]),
  p_value = co[, "Pr(>|z|)"]
)
or <- or[or$term != "(Intercept)", ]
or <- or[order(or$p_value), ]

auroc <- function(y, p) {
  r <- rank(p); n1 <- sum(y == 1); n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
test <- df[df$split == "test", ]
auc_nn  <- auroc(test$readmitted_30d, test$predicted_risk)
auc_glm <- auroc(test$readmitted_30d, predict(fit, test, type = "response"))

brk <- unique(quantile(test$predicted_risk, probs = seq(0, 1, 0.1)))
test$decile <- cut(test$predicted_risk, brk, include.lowest = TRUE, labels = FALSE)
calib <- aggregate(cbind(predicted = predicted_risk, observed = readmitted_30d) ~ decile,
                   data = test, FUN = mean)
calib$n <- format(as.vector(table(test$decile)), big.mark = ",")

md_table <- function(d, digits = 3) {
  d[] <- lapply(d, function(x) if (is.numeric(x)) formatC(x, digits = digits, format = "g") else as.character(x))
  c(paste("|", paste(names(d), collapse = " | "), "|"),
    paste("|", paste(rep("---", ncol(d)), collapse = " | "), "|"),
    apply(d, 1, function(r) paste("|", paste(r, collapse = " | "), "|")))
}

out <- c(
  "# CareSignal model validation (R)", "",
  sprintf("Data directory: `%s`. Records: %d; held-out test records: %d.", art, nrow(df), nrow(test)), "",
  sprintf("Features used (%d): %s", length(features), paste(features, collapse = ", ")), "",
  "## Held-out discrimination (AUROC)", "",
  sprintf("- PyTorch neural net: **%.3f**", auc_nn),
  sprintf("- Logistic regression baseline: **%.3f**", auc_glm), "",
  "## Calibration of the neural net by risk decile (test set)", "",
  md_table(calib), "",
  "## Logistic regression odds ratios (training set)", "",
  md_table(or)
)
writeLines(out, file.path(art, "r_validation.md"))
cat(sprintf("AUROC  neural net %.3f | logistic baseline %.3f\n", auc_nn, auc_glm))
cat("Wrote", file.path(art, "r_validation.md"), "\n")
