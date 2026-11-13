#!/usr/bin/env Rscript
# Independent statistical validation of the PyTorch risk model (base R only).
#   Rscript r/analysis.R            -> writes artifacts/r_validation.md
# 1. Logistic-regression baseline with odds ratios and 95% CIs.
# 2. Held-out discrimination: neural net vs. the GLM baseline (AUROC).
# 3. Calibration of the neural net by predicted-risk decile.

art <- "artifacts"
cohort <- read.csv(file.path(art, "cohort.csv"), stringsAsFactors = FALSE)
preds  <- read.csv(file.path(art, "predictions.csv"), stringsAsFactors = FALSE)
df <- merge(cohort, preds, by = "patient_id")

# Same imputation rule as the Python pipeline: training-set medians.
train <- df$split == "train"
for (lab in c("hba1c", "creatinine")) {
  df[[lab]][is.na(df[[lab]])] <- median(df[[lab]][train], na.rm = TRUE)
}
df$discharge_disposition <- relevel(factor(df$discharge_disposition), ref = "home")

fit <- glm(
  readmitted_30d ~ age + sex + length_of_stay + prior_admissions_12m +
    num_medications + charlson_index + hba1c + creatinine + systolic_bp + bmi +
    chf + copd + diabetes + ckd + lives_alone + discharge_disposition,
  data = df[train, ], family = binomial()
)

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

test$decile <- cut(test$predicted_risk,
                   quantile(test$predicted_risk, 0:10 / 10), include.lowest = TRUE, labels = 1:10)
calib <- aggregate(cbind(predicted = predicted_risk, observed = readmitted_30d) ~ decile, test, mean)
calib$n <- as.vector(table(test$decile))

md_table <- function(d, digits = 3) {
  d[] <- lapply(d, function(x) if (is.numeric(x)) formatC(x, digits = digits, format = "g") else as.character(x))
  c(paste("|", paste(names(d), collapse = " | "), "|"),
    paste("|", paste(rep("---", ncol(d)), collapse = " | "), "|"),
    apply(d, 1, function(r) paste("|", paste(r, collapse = " | "), "|")))
}

out <- c(
  "# CareSignal model validation (R)", "",
  sprintf("Synthetic cohort: %d patients, %d held-out test patients.", nrow(df), nrow(test)), "",
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
