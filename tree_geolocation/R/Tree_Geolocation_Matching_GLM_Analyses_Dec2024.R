# Experiments with Mixed Effects Models for Tree Geolocation Experiments

# Load libraries
library(ggplot2)
library(lattice)
library(Matrix)
library(lme4)
library(afex)
library(merTools)
library(bbmle)
library(multcomp)



# Read results of tree geolocation experiment across cities
trees <- read.csv("Z:/auto_arborist_cvpr2022_v0.15/analyses/tree_geolocation/all_city_filtered_geolocation_results_greedy_matching_tables_figures/all_cities_filtered_output_tree_greedy_genus_matching_results_jan625.csv")

# Format data types
trees$City <- factor(trees$City)
trees$inventory_genus_name <- factor(trees$inventory_genus_name)
trees$grid_density_level <- factor(trees$grid_density_level, levels = c("Low Density", "Medium Density", "High Density"))

# Running a basic logistic regression (binomial glm) with no random effects (using stats)
glm1 <- glm(proportion_matches ~ grid_num_panos + min_nn_distance + num_inv_trees,
            data = trees, family = "binomial", weights = num_inv_trees)

# Fixed effect estimates look OK 
print(glm1)
summary(glm1)

# Drop variables from the model and check AIC/ Deviance changes
drop1(glm1, test = "Chi")

# Running a logistic regression (binomial glm) with random City intercept effect
glmer1 <- glmer(proportion_matches ~ grid_num_panos + min_nn_distance + num_inv_trees + (1 | City) + (1 | inventory_genus_name),
              data = trees, family = "binomial", weights = num_inv_trees, control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))

# GLMER1 has a convergence issue - center and scale predictor variables
trees.CS.1 <- transform(trees, grid_num_panos_cs=scale(grid_num_panos))
trees.CS.2 <- transform(trees.CS.1, min_nn_distance_cs=scale(min_nn_distance))
trees.CS.preds <- transform(trees.CS.2, num_inv_trees_cs=scale(num_inv_trees))

# Convert the response into a two-colum matrix (successes, failures) to model as a binomial variable
# Proportion_matches - a fraction p = m / n of two integers. m is num_genus_matches. n is num_inv_trees

# Logistic regression with random intercepts and slopes 
trees.CS.preds$successes <- trees$num_genus_matches
trees.CS.preds$failures <- trees$num_inv_trees - trees$num_genus_matches

# Inspect rescaled variables
head(trees.CS.preds)

# Running a logistic regression (binomial glm) with random City and Genus intercept effects and binomial response and rescaled variables

# GLMER using tree density as counts
glmer2 <- glmer(cbind(successes, failures) ~ grid_num_panos_cs + min_nn_distance_cs + num_inv_trees_cs + (1 | City) + (1 | inventory_genus_name),
                  data = trees.CS.preds, family = "binomial", control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))

# GLMER using tree density as categorical (density levels)
glmer2.1 <- glmer(cbind(successes, failures) ~ grid_num_panos_cs + min_nn_distance_cs + grid_density_level + (1 | City) + (1 | inventory_genus_name),
                data = trees.CS.preds, family = "binomial", control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))

# GLMER fit using AFEX R package to calculate p-values for all fixed effects
glmer2.2 <- mixed(cbind(successes, failures) ~ grid_num_panos_cs * min_nn_distance_cs * num_inv_trees_cs + (1 | City) + (1 | inventory_genus_name),
                  data = trees.CS.preds, family = "binomial", control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)), method = "PB")

# Drop variables from the model and check AIC/ Deviance changes
drop1(glmer2, test = "Chi")

# Inspect fixed and random effect coefficients
# Fixed effect estimates
print(glmer2)
summary(glmer2)
# Random effect of city intercept
print(ranef(glmer2))

# Simulate and plot fixed effects estimates
feEx <- FEsim(glmer2, 1000)
plotFEsim(feEx) + theme_bw() + labs(title = "Fixed Effects Coefficient Plot",
                                    x = "Median Effect Estimate", y = "Model Term")

# Simulate and plot random effects (City) estimates
reEx <- REsim(glmer2)
lattice::dotplot(ranef(glmer2, condVar=TRUE))[1] # Random effect of genus
lattice::dotplot(ranef(glmer2, condVar=TRUE))[2] # Random effect of city

# Post-Hoc Tests General LInear Hypotheses for Grid Density Levels
# Example: Pairwise comparisons for grid_density_levels
posthoc_grid_density <- glht(glmer2.1, linfct = mcp(grid_density_level = "Tukey"))
summary(posthoc_grid_density)  # Show results
confint(posthoc_grid_density)  # Confidence intervals
plot(posthoc_grid_density)     # Visualization








# Combinations of models with varying random intercepts and fixed effects to find best fit based on AIC

# Single predictor + (1 | City)
model1 <- glmer(cbind(successes, failures) ~ grid_num_panos_cs + (1 | City), 
                data = trees.CS.preds, family = "binomial", control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))
model2 <- glmer(cbind(successes, failures) ~ min_nn_distance_cs + (1 | City), 
                data = trees.CS.preds, family = "binomial", control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))
model3 <- glmer(cbind(successes, failures) ~ num_inv_trees_cs + (1 | City), 
                data = trees.CS.preds, family = "binomial", control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))

# Single predictor + (1 | inventory_genus_name)
model4 <- glmer(cbind(successes, failures) ~ grid_num_panos_cs + (1 | inventory_genus_name), 
                data = trees.CS.preds, family = "binomial", control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))
model5 <- glmer(cbind(successes, failures) ~ min_nn_distance_cs + (1 | inventory_genus_name), 
                data = trees.CS.preds, family = "binomial", control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))
model6 <- glmer(cbind(successes, failures) ~ num_inv_trees_cs + (1 | inventory_genus_name), 
                data = trees.CS.preds, family = "binomial", control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))

# Single predictor + (1 | City) + (1 | inventory_genus_name)
model7 <- glmer(cbind(successes, failures) ~ grid_num_panos_cs + (1 | City) + (1 | inventory_genus_name), 
                data = trees.CS.preds, family = "binomial", control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))
model8 <- glmer(cbind(successes, failures) ~ min_nn_distance_cs + (1 | City) + (1 | inventory_genus_name), 
                data = trees.CS.preds, family = "binomial", control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))
model9 <- glmer(cbind(successes, failures) ~ num_inv_trees_cs + (1 | City) + (1 | inventory_genus_name), 
                data = trees.CS.preds, family = "binomial", control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))

# Two predictors + (1 | City)
model10 <- glmer(cbind(successes, failures) ~ grid_num_panos_cs + min_nn_distance_cs + (1 | City), 
                 data=trees.CS.preds, family="binomial", control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))
model11 <- glmer(cbind(successes, failures) ~ grid_num_panos_cs + num_inv_trees_cs + (1 | City), 
                 data=trees.CS.preds, family="binomial", control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))
model12 <- glmer(cbind(successes, failures) ~ min_nn_distance_cs + num_inv_trees_cs + (1 | City), 
                 data=trees.CS.preds, family="binomial", control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))

# Two predictors + (1 | inventory_genus_name)
model13 <- glmer(cbind(successes, failures) ~ grid_num_panos_cs + min_nn_distance_cs + (1 | inventory_genus_name), 
                 data=trees.CS.preds, family="binomial", control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))
model14 <- glmer(cbind(successes, failures) ~ grid_num_panos_cs + num_inv_trees_cs + (1 | inventory_genus_name), 
                 data=trees.CS.preds, family="binomial", control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))
model15 <- glmer(cbind(successes, failures) ~ min_nn_distance_cs + num_inv_trees_cs + (1 | inventory_genus_name), 
                 data=trees.CS.preds, family="binomial", control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))

# Two predictors + (1 | City) + (1 | inventory_genus_name)
model16 <- glmer(cbind(successes, failures) ~ grid_num_panos_cs + min_nn_distance_cs + (1 | City) + (1 | inventory_genus_name), 
                 data=trees.CS.preds, family="binomial", control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))
model17 <- glmer(cbind(successes, failures) ~ grid_num_panos_cs + num_inv_trees_cs + (1 | City) + (1 | inventory_genus_name), 
                 data=trees.CS.preds, family="binomial", control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))
model18 <- glmer(cbind(successes, failures) ~ min_nn_distance_cs + num_inv_trees_cs + (1 | City) + (1 | inventory_genus_name), 
                 data=trees.CS.preds, family="binomial", control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))

# Three predictors + (1 | City)
model19 <- glmer(cbind(successes, failures) ~ grid_num_panos_cs + min_nn_distance_cs + num_inv_trees_cs + (1 | City), 
                 data=trees.CS.preds, family="binomial", control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))

# Three predictors + (1 | inventory_genus_name)
model20 <- glmer(cbind(successes, failures) ~ grid_num_panos_cs + min_nn_distance_cs + num_inv_trees_cs + (1 | inventory_genus_name), 
                 data=trees.CS.preds, family="binomial", control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))

# Three predictors + (1 | City) + (1 | inventory_genus_name)
model21 <- glmer(cbind(successes, failures) ~ grid_num_panos_cs + min_nn_distance_cs + num_inv_trees_cs + (1 | City) + (1 | inventory_genus_name), 
                 data=trees.CS.preds, family="binomial", control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))

# Three predictors + (1 | City) + (1 | inventory_genus_name)
model22 <- glmer(cbind(successes, failures) ~ grid_num_panos_cs * min_nn_distance_cs * num_inv_trees_cs + (1 | City) + (1 | inventory_genus_name), 
                 data=trees.CS.preds, family="binomial", control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)))


# Compare models by AIC
AICtab(model1, model2, model3, model4, model5, model6, model7, model8, model9,
       model10, model11, model12, model13, model14, model15, model16, model17, model18,
       model19, model20, model21, model22, weights = TRUE)
































































