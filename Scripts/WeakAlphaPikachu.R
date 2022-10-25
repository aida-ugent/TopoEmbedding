# The purpose of this tutorial is to introduce persistent homology in R.
# We first set the working directory and import the necessary libraries.

setwd("..") # change working directory to main repository 
library("TDA") # persistent homology in R
library("ggplot2") # plotting
library("latex2exp") # LaTeX text in figures
library("gridExtra") # arrange multiple plots

# We load and plot a point cloud data set representing Pikachu.

df <- read.table(file.path("Pikachu.csv"), sep=",", col.names=c("x", "y"))
ggplot(df, aes(x=x, y=y)) +
  geom_point(size=1.5) +
  coord_fixed() +
  theme_bw() +
  theme(text=element_text(size=40))
ggsave("Pikachu.png", width=800, height = 1000, unit='px', dpi = 100)

# The weak alpha filtration is not implemented in R.
# Thus, we obtain it by modifying the filtration times in the regular alpha filtration.
# This is just for illustration, and not recommended for general applications.

filtration <- alphaComplexFiltration(df)
new_values <- sapply(filtration$cmplx, function(s){
  if(length(s) == 1) return(0)
  else return(max(dist(df[s,])))
})
Idx <- order(new_values)
filtration$values <- new_values[Idx]
filtration$cmplx <- filtration$cmplx[Idx] 

# To illustrate the weak alpha filtration, we visualize the simplicial complexes for a few (increasing) time/alpha parameters.

alphas <- c(0, 5, 10, 15, 20, Inf) # time/alpha parameters at which to show the simplicial complex

simpPlots <- list()
for(idx in 1:length(alphas)){
  print(paste("Constructing plot for time parameter alpha =", alphas[idx]))
  
  # Initialize the complex
  if(idx == 1){
    edges <- data.frame(x1=numeric(0), y1=numeric(0), x2=numeric(0), y2=numeric(0))
    triangles  <- data.frame(id=integer(0), x=numeric(0), y=numeric(0))
  }
  
  # Determine new simplices for this complex
  this_simplices_I <- which(filtration$values <= alphas[idx] &
                              filtration$values > max(alphas[idx - 1], 0))
  
  # Add new simplices to the the plot, if any
  if (length(this_simplices_I) > 0){
    
    # Determine and add new edges for this complex
    this_edges_I <- which(sapply(filtration$cmplx[this_simplices_I], function(s) length(s) == 2))
    if(length(this_edges_I) > 0){
      this_edges <- matrix(do.call("rbind", filtration$cmplx[this_simplices_I[this_edges_I]]), ncol=2)
      this_edges <- data.frame(cbind(df[this_edges[,1], c("x", "y")], df[this_edges[,2], c("x", "y")]))
      colnames(this_edges) <- c("x1", "y1", "x2", "y2")
      edges <- rbind(edges, this_edges)
    }
    
    # Determine and add new triangles for this complex
    this_triangles_I <- which(sapply(filtration$cmplx[this_simplices_I], function(s) length(s) == 3))
    if(length(this_triangles_I) > 0){
      this_triangles_vertex_I <- unlist(filtration$cmplx[this_simplices_I[this_triangles_I]])
      new_triangles_grouping <- rep((nrow(triangles) / 3 + 1):
                                      (nrow(triangles) / 3 + length(this_triangles_I)), each=3)
      triangles <- rbind(triangles, cbind(id=new_triangles_grouping, df[this_triangles_vertex_I, c("x", "y")]))
    }
  }
  
  # Plot the simplicial complex
  simpPlots[[length(simpPlots) + 1]] <- ggplot() +
    geom_polygon(data=triangles, aes(x=x, y=y, group=id), fill="green", alpha=0.75) +
    geom_segment(data=edges, aes(x=x1, y=y1, xend=x2, yend=y2), color="black", size=0.5, alpha=0.75) +
    {if(nrow(edges) == 0) geom_point(data=df, aes(x=x, y=y), size=1.5)
      else geom_point(data=df, aes(x=x, y=y), size=1.5, alpha=0.75)} +
    coord_fixed() +
    theme_bw() +
    ggtitle(TeX(sprintf("$\\alpha = %g$", alphas[idx]))) +
    theme(plot.title=element_text(hjust=0.5, size=50), text=element_text(size=40))
}

gridplot <- grid.arrange(grobs=simpPlots, nrow=2, ncol=3)
ggsave("PikachuFiltration.png", 
       height=2000, width=2500, units='px', dpi=100, plot=gridplot)

pltalpha <- simpPlots[6][[1]] + ggtitle(element_blank())
ggsave("PikachuAlpha.png", 
       height=1000, width=800, units='px', dpi=100)

# We see that various connected components (0-dim. holes) and cycles (1-dim. holes) appear and disappear across this filtration.
# The idea is that holes that persist for longer correspond to siginificant features of the underlying topological model.
# We can compute the persistence of these features as follows.

diag <- filtrationDiag(filtration, maxdimension=1)

# We can visualize the results of persistent homology through a persistence diagram.
# In the diagram, a point (b, d) represent a topological hole/feature that persists from time b to d.
# We also encircle the points in the diagrams corresponding to Pikachu's basic topological features. 

lim <- 25 # upper limit for plotting persistence diagram points
components <- diag[["diagram"]][diag[["diagram"]][,1] == 0 & diag[["diagram"]][,2] < 5,][1:5,2:3]
components[1, 2] <- lim # cannot plot infinity, so choose a large number
circles <- diag[["diagram"]][diag[["diagram"]][,1] == 1 & diag[["diagram"]][,2] < 5,][1:8,2:3]
op <- par(mar=c(4.25, 4.75, 1, 1))
diagplot <- plot(NULL, xlim=c(0, lim), ylim=c(0, lim), xlab="Birth", ylab="Death", cex.lab=2, cex.axis=2)
points(components[,1], components[,2], cex=2.5, col="black")
points(circles[,1], circles[,2], cex=2.5, col="red")
plot.diagram(diag[["diagram"]], diagLim=c(0, lim), dim=1, add=TRUE, col="red")
plot.diagram(diag[["diagram"]], diagLim=c(0, lim), dim=0, add=TRUE, col="black")
abline(h=lim, lty=2) # line marking infinite death time
legend(x=17.5, y=12.5, legend=c("H0", "H1"), col=c("black", "red"), 
       pch=c(19, 2), pt.lwd=2, box.lty=0, cex=2); par(op)