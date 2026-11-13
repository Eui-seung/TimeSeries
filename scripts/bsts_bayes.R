# 베이지안 BSTS (R bsts): MCMC로 사후분포 표본을 뽑아 CSV로 저장. scripts/bsts_bayes.py가 호출함.
#
# 사용: Rscript bsts_bayes.R <입력 csv> <출력 접두어> <niter> <burn> <thin> <seed>
#   입력 csv: date, y(log total, 가린 날은 빈칸), 회귀 변수들 (학습 기간 2014~2023만)
#   출력: <접두어>_draws.csv  사후분포 표본 (statsmodels UnobservedComponents 모수 이름과 단위로 변환)
#         <접두어>_coef.csv   회귀 계수 사후 평균·표준편차·포함 확률
#
# 모형: local level + AR(1) + 주간 계절성(주기 7, 더미형) + 회귀 (bsts_model.py의 BSTS와 같은 구조)
# 회귀 사전분포: spike-and-slab. 통제 변수(푸리에·계단 더미·여름휴가)는 포함 확률 1로 고정,
#   날씨 변수(이름이 heat_/cold_/rain/snow로 시작)는 포함 확률 0.5에서 데이터로 고름
# 주의: 시계열이 NA로 시작하면 bsts의 사전분포 계산이 깨지므로(mu = NaN 오류) 첫 관측일부터 넣음

suppressMessages(library(bsts))
a <- commandArgs(trailingOnly = TRUE)
in_csv <- a[1]; out <- a[2]
niter <- as.integer(a[3]); burn <- as.integer(a[4]); thin <- as.integer(a[5]); seed <- as.integer(a[6])

d <- read.csv(in_csv, check.names = FALSE)
first <- which(!is.na(d$y))[1]
d <- d[first:nrow(d), ]
y <- d$y
X <- as.matrix(d[, setdiff(names(d), c('date', 'y')), drop = FALSE])
sdy <- sd(y, na.rm = TRUE)
obs <- !is.na(y)

ss <- AddLocalLevel(list(), y, sdy = sdy)
ss <- AddAr(ss, y, lags = 1, sdy = sdy)
ss <- AddSeasonal(ss, y, nseasons = 7, sdy = sdy)
weather <- grepl('^(heat_|cold_|rain|snow)', colnames(X))
prior <- SpikeSlabPrior(X[obs, , drop = FALSE], y[obs], prior.inclusion.probabilities = ifelse(weather, 0.5, 1))

t0 <- Sys.time()
m <- bsts(y ~ X - 1, ss, niter = niter, ping = 0, seed = seed, prior = prior, data = list(y = y, X = X))
secs <- as.numeric(Sys.time() - t0, units = 'secs')

k <- seq(burn + 1, niter, by = thin)
beta <- m$coefficients[k, , drop = FALSE]
colnames(beta) <- paste0('beta.', sub('^X', '', colnames(beta)))
draws <- data.frame(
  sigma2.irregular = m$sigma.obs[k]^2,
  sigma2.level = m$sigma.level[k]^2,
  sigma2.seasonal = m$sigma.seasonal.7[k]^2,
  sigma2.ar = m$AR1.sigma[k]^2,
  ar.L1 = m$AR1.coefficients[k],
  beta, check.names = FALSE)
write.csv(draws, paste0(out, '_draws.csv'), row.names = FALSE)

coef <- data.frame(var = sub('^beta\\.', '', colnames(beta)), weather = weather,
                   mean = colMeans(beta), sd = apply(beta, 2, sd), incl_prob = colMeans(beta != 0),
                   mean_if_included = sapply(seq_len(ncol(beta)), function(j) { b <- beta[, j]; if (any(b != 0)) mean(b[b != 0]) else 0 }))
write.csv(coef, paste0(out, '_coef.csv'), row.names = FALSE)
cat(sprintf('done %s  draws=%d  %.1f s\n', basename(out), length(k), secs))
