# This script shows how noise in high dimensions may affect the PCA embedding.
# We first import the necessary libraries.

library("ggplot2") # plotting
library("gridExtra") # arrange multiple plots
library("circular") # circular colors

# We construct a circular data set and sample noise in high dimensions

npoints <- 100 # number of points
dims <- c(2, 500, 1000, 2500, 5000) # dimensions of interest
maxdim <- max(dims)
sigma <- 0.25 # sd of noise
noisetype <- c("Uniform", "Gaussian")[1]
t <- seq(-pi, pi, length.out=npoints)
col <- circular.colors(npoints)
X <- data.frame(x=cos(t), y=sin(t))
set.seed(42)
if(noisetype=="Uniform") # cubical noise
  N <- matrix(sqrt(3 * sigma**2) * 2 * (runif(npoints * maxdim) - 1 / 2), nrow=npoints, ncol=maxdim)
if(noisetype=="Gaussian") # spherical noise
  N <- matrix(rnorm(npoints * maxdim, sd=sigma), nrow=npoints, ncol=maxdim) # spherical noise
df <- cbind(X, matrix(numeric(npoints * (maxdim - 2)), nrow=npoints, ncol=maxdim - 2)) + N

# We view the PCA embedding obtained from various data dimensionalities

plotlist <- list()
for(dim in dims){
  pca <- data.frame(prcomp(df[,1:dim], rank.=2)$x)
  angles <- atan2(pca[,2], pca[,1])
  anglesflip <- atan2(pca[,2], -pca[,1])
  if(sd(pmin(abs(angles - t), abs(angles + 2 * pi - t), abs(angles - 2 * pi - t))) < 
     sd(pmin(abs(anglesflip - t), abs(anglesflip + 2 * pi - t), abs(anglesflip - 2 * pi - t))))
    pca[,1] <- -pca[,1] # flip for consistency
  plotlist[[length(plotlist) + 1]] <- ggplot(pca, aes(x=PC1, y=PC2)) +
    geom_point(size=1.5, col=col) +
    theme_bw() +
    ggtitle(paste("dim =", dim)) +
    theme(plot.title=element_text(hjust=0.5))
}

gridplot <- grid.arrange(grobs=plotlist, nrow=1, ncol=5)
ggsave(filename=paste0("NoisyBall", noisetype, ".pdf"), plot=gridplot,
       width=1100, height=235, units='px', dpi=100)
