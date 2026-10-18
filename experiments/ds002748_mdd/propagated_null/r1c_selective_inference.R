#R side of r1c: runs both selective-inference methods on a cell from r1c_selective_inference.py
#failures are recorded, not dropped
#usage: Rscript r1c_selective_inference.R <cell_dir>

suppressPackageStartupMessages({
  library(clusterpval)
  library(KmeansInference)
})

args <- commandArgs(trailingOnly = TRUE)
cell_dir <- args[1]
manifest <- jsonlite::fromJSON(file.path(cell_dir, "manifest.json"))

n <- manifest$n
d <- manifest$d
N <- manifest$n_datasets
K <- manifest$K
con <- file(file.path(cell_dir, "data.bin"), "rb")
raw <- readBin(con, "double", n = N * n * d, size = 8, endian = "little")
close(con)

#pooled isotropic sigma, the packages' own default
pooled_sig <- function(X) {
  q <- ncol(X)
  sqrt(sum(scale(X, scale = FALSE)^2) / (nrow(X) * q - q))
}

rows <- vector("list", N)
for (i in seq_len(N)) {
  off <- (i - 1) * n * d
  X <- matrix(raw[(off + 1):(off + n * d)], nrow = n, ncol = d, byrow = TRUE)

  rec <- list(dataset = i - 1,
              p_hier_iso = NA_real_, p_hier_gencov = NA_real_,
              p_kmeans_iso = NA_real_, p_kmeans_gencov = NA_real_,
              sig_hat = NA_real_,
              err_hier_iso = "", err_hier_gencov = "",
              err_kmeans_iso = "", err_kmeans_gencov = "")

  rec$sig_hat <- tryCatch(pooled_sig(X), error = function(e) NA_real_)

  #clusterpval, ward, isotropic
  rec$p_hier_iso <- tryCatch({
    hcl <- hclust(dist(X)^2, method = "ward.D")
    test_hier_clusters_exact(X, link = "ward.D", hcl = hcl, K = K,
                             k1 = 1, k2 = 2, iso = TRUE, sig = NULL)$pval
  }, error = function(e) { rec$err_hier_iso <<- conditionMessage(e); NA_real_ })

  #general covariance; fails when n <= d since cov(X) is singular
  rec$p_hier_gencov <- tryCatch({
    hcl <- hclust(dist(X)^2, method = "ward.D")
    test_hier_clusters_exact(X, link = "ward.D", hcl = hcl, K = K,
                             k1 = 1, k2 = 2, iso = FALSE)$pval
  }, error = function(e) { rec$err_hier_gencov <<- conditionMessage(e); NA_real_ })

  #KmeansInference, isotropic, pooled sigma
  rec$p_kmeans_iso <- tryCatch({
    kmeans_inference(X, k = K, 1, 2, iso = TRUE, sig = rec$sig_hat,
                     iter.max = 20, seed = 20260831 + i, verbose = FALSE)$pval
  }, error = function(e) { rec$err_kmeans_iso <<- conditionMessage(e); NA_real_ })

  #KmeansInference, general covariance
  rec$p_kmeans_gencov <- tryCatch({
    kmeans_inference(X, k = K, 1, 2, iso = FALSE,
                     iter.max = 20, seed = 20260831 + i, verbose = FALSE)$pval
  }, error = function(e) { rec$err_kmeans_gencov <<- conditionMessage(e); NA_real_ })

  rows[[i]] <- as.data.frame(rec, stringsAsFactors = FALSE)
}

out <- do.call(rbind, rows)
write.csv(out, file.path(cell_dir, "results.csv"), row.names = FALSE)

cat(sprintf("cell %s: %d datasets\n", basename(cell_dir), N))
for (col in c("p_hier_iso", "p_hier_gencov", "p_kmeans_iso", "p_kmeans_gencov")) {
  ok <- sum(!is.na(out[[col]]))
  rej <- if (ok > 0) mean(out[[col]][!is.na(out[[col]])] < 0.05) else NA
  cat(sprintf("  %-16s ran %4d/%d  rejection %s\n", col, ok, N,
              if (is.na(rej)) "n/a" else sprintf("%.4f", rej)))
}
