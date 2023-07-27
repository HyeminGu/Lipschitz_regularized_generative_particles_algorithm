# Version info: R 3.2.3, Biobase 2.30.0, GEOquery 2.40.0, limma 3.26.8
################################################################
# install packages
install.packages("BiocManager")
BiocManager::install("GEOquery")
install.packages("umap")

#   Data plots for selected GEO samples
library(GEOquery)
library(limma)
library(umap)

# load series and platform data from GEO
# Breast cancer
folder <- "Labeled_disease"
accession_code <- "GSE26639" # GSE76275
gset <- getGEO(accession_code, GSEMatrix =TRUE, getGPL=FALSE)

if (length(gset) > 1) idx <- grep("GPL570", attr(gset, "names")) else idx <- 1
gset <- gset[[idx]]

ex <- exprs(gset)
# log2 transform
qx <- as.numeric(quantile(ex, c(0., 0.25, 0.5, 0.75, 0.99, 1.0), na.rm=T))
LogC <- (qx[5] > 100) || (qx[6]-qx[1] > 50 && qx[2] > 0)
if (LogC) {
  ex[which(ex <= 0)] <- NaN
  ex <- log2(ex)
  }

## box-and-whisker plot
#dev.new(width=3+ncol(gset)/6, height=5)
#par(mar=c(7,4,2,1))
#title <- paste ("GSE47109", "/", annotation(gset), sep ="")
#boxplot(ex, boxwex=0.7, notch=T, main=title, outline=FALSE, las=2)
#dev.off()

## expression value distribution plot
#par(mar=c(4,4,2,1))
#title <- paste ("GSE47109", "/", annotation(gset), " value distribution", sep ="")
#plotDensities(ex, main=title, legend=F)

# mean-variance trend
ex <- na.omit(ex) # eliminate rows with NAs
#plotSA(lmFit(ex), main="Mean variance trend, GSE47109")

# UMAP plot (multi-dimensional scaling)
ex <- ex[!duplicated(ex), ]  # remove duplicates
#ump <- umap(t(ex), n_neighbors = 15, random_state = 123)
#plot(ump$layout, main="UMAP plot, nbrs=15", xlab="", ylab="", pch=20, cex=1.5)
#library("maptools")  # point labels without overlaps
#pointLabel(ump$layout, labels = rownames(ump$layout), method="SANN", cex=0.6)

# save gene expression values to csv
#write.matrix(ex, file="/Users/pantazis/Downloads/gene_expression_bio_example/GPL570/GSE47109.csv", sep = ",")
#write.matrix(ex, file="/Users/pantazis/Downloads/gene_expression_bio_example/GPL570/GSE10843.csv", sep = ",")
#write.matrix(ex, file="/Users/pantazis/Downloads/gene_expression_bio_example/GPL570/GSE102484.csv", sep = ",")

#write.matrix(ex, file="/Users/pantazis/Downloads/gene_expression_bio_example/GPL570/GSE6891.csv", sep = ",")
#write.matrix(ex, file="/Users/pantazis/Downloads/gene_expression_bio_example/GPL570/GSE61804.csv", sep = ",")

#write.matrix(ex, file="/Users/pantazis/Downloads/gene_expression_bio_example/GPL570/GSE11882.csv", sep = ",")
write.matrix(ex, file=sprintf("./GPL570/%s/%s.csv",folder, accession_code), sep = ",")
