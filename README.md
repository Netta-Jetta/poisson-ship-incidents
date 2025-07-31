# Poisson Regression: Ship Incident Modeling  
**CU Boulder**: *Statistical Modeling for Data Science Applications – Peer Reviewed Assignment*

<br>

<h2><img width="20" height="20" alt="image" src="https://github.com/user-attachments/assets/8aa305af-695c-4973-882d-9d6ad9dd49ef" /> Overview</h2>

This project models ship damage incidents using **Poisson regression**.  
I compare **full and reduced models**, evaluate predictive accuracy using **MSPE**, check for **overdispersion**, and visualize key **diagnostics** to assess model fit.

---

<h2><img src="https://github.com/user-attachments/assets/d9d74e78-71d5-40fd-ae15-6070a5d6f2c8" width="20" height="20" alt="icon" /> Problem</h2>

I'm analyzing a dataset of ship damage incidents. Each ship type was observed over a number of months, and I want to understand what factors contribute to more (or fewer) incidents. Since the outcome is **count data** relative to time (service months), **Poisson regression** is appropriate.

---

<h2><img width="20" height="20" alt="image" src="https://github.com/user-attachments/assets/88e372d9-5d48-4900-a00f-55d4295b7ee6" /> Data Summary</h2>  

Variables include:

- `incidents`: Number of damage incidents (count)
- `service`: Total months of service (exposure)
- `type`: Ship type (categorical)
- `year`: Year built (categorical)
- `period`: Operation period (categorical)

---

<h2><img width="20" height="20" alt="image" src="https://github.com/user-attachments/assets/56d57607-fbf1-4145-86af-7e07f9cd8b77" /> Setup</h2>

To run this analysis in R, you’ll need the following packages:

- `stats` (base)
- `MASS`
- `ggplot2` (if using optional visualizations)

---

<h2><img width="20" height="20" alt="image" src="https://github.com/user-attachments/assets/74e75c95-b85d-46cd-93f6-e96830813714" />
Model 1: Full Poisson Model</h2>

I fit a Poisson model using all predictors:

```r
full_model <- glm(incidents ~ type + period + year,
             family = poisson,
             data = train,
             offset = log(service))
```

### Why offset `log(service)`?

To convert the model from a *count* to a *rate* model (incidents per month).

### Model Output Summary

<img src="model_summary.png" alt="Full Model - GLM Summary Output" width="600"/>

This summary shows the coefficients, standard errors, and significance levels for each predictor in the full Poisson model.

### MSPE (Full Model)

```r
mspe_full <- mean((test$incidents - predict(full_model, newdata = test, type = "response"))^2)
# MSPE = 15.26
```

The **Mean Squared Prediction Error (MSPE)** for the full model is **15.26** on the test set.

---

<h2><img width="20" height="20" alt="image" src="https://github.com/user-attachments/assets/74e75c95-b85d-46cd-93f6-e96830813714" /> Model 2: Reduced Poisson Model</h2>

I dropped `year` from the model to test whether a simpler version might generalize better:

```r
model_reduced <- glm(incidents ~ type + period,
                     family = poisson,
                     data = train,
                     offset = log(service))
```

### MSPE (Reduced Model)

```r
mspe_reduced <- mean((test$incidents - predict(model_reduced, newdata = test, type = "response"))^2)
# MSPE = 65.20
```

*The MSPE increased significantly to **65.20**, indicating that dropping `year` reduced predictive performance.*

---
<h2><img width="20" height="20" alt="image" src="https://github.com/user-attachments/assets/f7bb18a2-2d5d-43ae-8545-9cd4fe67c8dc" />
MSPE: Full vs Reduced Model</h2>

<img src="MSPE_comparison.png" alt="MSPE Comparison" width="500"/>

The bar chart above summarizes the MSPEs for both models, clearly showing that the **full model** outperforms the reduced one.

<img width="20" height="20" alt="checkmark" src="https://github.com/user-attachments/assets/a464f70a-aaae-4578-a2f2-6c9f8d2bc479" /> **Conclusion:** Including `year` improves predictive accuracy and should be retained in the model.

---

<h2><img width="20" height="20" alt="image" src="https://github.com/user-attachments/assets/7458e3fc-1eca-4a2d-8aa1-9bf71a675e65" /> Deviance Testing</h2>

I compared models using deviance to assess model fit:

### Null vs Full:

```r
anova(null_model, full_model, test = "Chisq")
# p < 0.001 → Full model is a huge improvement
```

### Full vs Reduced:

```r
anova(full_model, reduced_model, test = "Chisq")
# p < 0.001 → Removing `year` significantly worsens fit
```

<img width="20" height="20" alt="image" src="https://github.com/user-attachments/assets/0b6121c8-c01a-4870-918d-bfe5717ee219" />
 I chose to stick with the full model.

---

<h2><img width="20" height="20" alt="image" src="https://github.com/user-attachments/assets/eba2f88d-e13e-4696-8b84-53759cf6e255" />
Residuals vs Linear Predictor</h2>

<img src="Residuals.png" alt="Residuals Plot" width="500"/>

**Residuals vs. Linear Predictor Plot**  
*Residuals are roughly symmetric and centered around 0 with no strong violations. This supports a good model fit.*

```r
plot(full_model$linear.predictors, residuals(full_model, type = "deviance"),
     xlab = "Linear Predictor", ylab = "Deviance Residuals",
     main = "Residuals vs Linear Predictor")
abline(h = 0, col = "red", lty = 2)
```

---

<h2><img width="20" height="20" alt="image" src="https://github.com/user-attachments/assets/40366a3e-9a66-4c96-a2d7-230f01f45ea4" />
Overdispersion Check</h2>

**Overdispersion Check**  
*A deviance/df ratio of 1.27 suggests mild overdispersion. May warrant quasi-Poisson if model complexity increases.*

```r
summary(full_model)$deviance / summary(full_model)$df.residual
# ≈ 1.27 → Slight overdispersion, but not severe
```

---

### Predicted vs Actual (Test Set)

<img src="observed_vs_predicted.png" alt="Observed vs Predicted" width="500"/>

```r
pred <- predict(full_model, newdata = test, type = "response")

plot(pred, test$incidents,
     xlab = "Predicted Incidents",
     ylab = "Observed Incidents",
     main = "Observed vs Predicted (Test Set)",
     pch = 19, col = "blue")
abline(0, 1, col = "red", lty = 2)
```

---
<h2><img width="20" height="20" alt="image" src="https://github.com/user-attachments/assets/1d93fbb8-025b-435f-8655-81ca338290a2" />
Distribution of Incidents by Ship Type</h2>

<img src="Incidents_by_Ship.png" alt="Incidents by Ship Type" width="500"/>

```r
boxplot(incidents ~ type, data = ships,
        main = "Incidents by Ship Type",
        xlab = "Ship Type", ylab = "Incident Count",
        col = "lightblue")
```

---

<h2><img width="20" height="20" alt="image" src="https://github.com/user-attachments/assets/79545368-27f2-4909-9a10-ab8e0aef55ab" />
Conclusions</h2>

- Poisson regression is a solid choice for modeling ship incident *rates*
- Including `year` improves predictive power
- The model fit is adequate (residual checks + deviance tests)
- Slight overdispersion is present but manageable

---

<h2><img width="72" height="72" alt="image" src="https://github.com/user-attachments/assets/7d83ce38-8928-44a8-9142-a7da70922413" />
Next Steps</h2>

While the full Poisson model performed well, there are opportunities to extend or refine the analysis:

- 🔁 **Explore interaction effects** — for example, `ship type × operation period` may reveal nuanced risk patterns.
- 🧠 **Test quasi-Poisson or negative binomial models** if further analysis reveals increased overdispersion.
- 🕒 **Explore grouping `year built` into eras** (e.g., pre-1970 vs post-1970) to test whether broader construction trends influence risk.
- 📈 **Validate the model on new datasets** or perform cross-validation to assess generalizability.

<br>

<br>
  
**Disclaimer**

*This project was completed as part of my personal learning through the open, non-credit version of the [Statistical Modeling for Data Science Applications](https://www.coursera.org/specializations/statistical-modeling) specialization by the University of Colorado Boulder on Coursera. All code and write-ups are my own work, and no proprietary content or solution materials from the for-credit program are included.*
