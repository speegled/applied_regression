set.seed(1262020)
x1s <- runif(30, 0, 10)
x2s <- runif(30, 0, 10)
x3s <- runif(30, 0, 10)
dd <- data.frame(x1s = x1s,
                 x2s = x2s,
                 x3s = x3s,
                 ys = 1 + 2 * x1s + 3 * x2s + rnorm(30, 0, 10) 
)

mod1 <- lm(ys ~ x1s, data = dd)
sigma_1 <- summary(mod1)$sigma
mod2 <- lm(ys ~ ., data = dd)
anova(mod1, mod2)
sigma_2 <- summary(mod2)$sigma
test_stat <- (28 * sigma_1^2 - (30 - (3 + 1))*sigma_2^2)/2/((30 - (3 + 1))*sigma_2^2) * 26
pf(test_stat, 2, 26, lower.tail = F)

sim_data <- replicate(1000, {
  dd$ys = 1 + 2 * x1s + rnorm(30, 0, 10) 
  mod1 <- lm(ys ~ x1s, data = dd)
  sigma_1 <- summary(mod1)$sigma
  mod2 <- lm(ys ~ ., data = dd)
  sigma_2 <- summary(mod2)$sigma
  (28 * sigma_1^2 - (30 - (3 + 1))*sigma_2^2)/2/((30 - (3 + 1))*sigma_2^2) * 26
})
plot(density(sim_data))
curve(df(x, 2, 26), add = T)

