# Clear the environment
rm(list = ls())

# Load required libraries
library(clusterProfiler)
library(org.Mm.eg.db)
library(pheatmap)
library(DESeq2)
library(limma)
library(ggpubr)
library(data.table)
library(Rcpp)
library(Seurat)
library(ggplot2)
library(reshape2)
library(dplyr)
library(grid)
library(CellChat)
library(ggalluvial)
library(NMF)

# Set working directory
setwd("~/stroke/AD_SCRNA/Result/CZX")

# Load Seurat object
PRO <- readRDS("~/stroke/AD_SCRNA/RDS/allcelltype_PRO.rds")

# Preprocess Seurat object
DefaultAssay(PRO) <- "RNA"
PRO@assays$RNA@data <- PRO@assays$RNA@counts
PRO <- NormalizeData(PRO)
PRO <- ScaleData(PRO, features = VariableFeatures(PRO))  # Scale top 2000 variable genes

# Define comparison for statistical tests
my_comparisons <- list(c("Control", "AD"))

# Function to plot gene expression violin plot by diagnosis
do_genevln <- function(gene_name) {
    TIMP1 <- as.data.frame(PRO@assays$RNA@data[which(rownames(PRO@assays$RNA@data) == gene_name), ])
    colnames(TIMP1) <- "Expression"
    TIMP1$barcode <- rownames(TIMP1)
    PRO@meta.data$barcode <- rownames(PRO@meta.data)
    TIMP_dat <- merge(PRO@meta.data, TIMP1, by = "barcode")
    rownames(TIMP_dat) <- TIMP_dat$barcode
    TIMP_dat <- TIMP_dat[, -1]  # Remove barcode column
    
    allcolour <- c("#0E98B2", "#F93F40")
    p <- ggplot(TIMP_dat, aes(x = diagnosis, y = Expression, fill = diagnosis)) +
        ggtitle(gene_name) +
        geom_violin(trim = TRUE, scale = "width") +
        geom_signif(comparisons = my_comparisons, map_signif_level = FALSE, test = wilcox.test,
                    y_position = max(TIMP_dat$Expression) - 0.5, textsize = 6) +
        stat_summary(aes(label = round(after_stat(y), 2)), fun = "mean", geom = "text", vjust = -1.5) +
        scale_fill_manual(values = allcolour) +
        theme_bw() +
        theme(panel.grid.major = element_blank(),
              panel.grid.minor = element_blank(),
              axis.text.x = element_text(angle = 45, hjust = 1)) +
        labs(y = "Expression Level")
    print(p)
}

# Function to plot gene expression violin plot by Braak stage
do_genevln_braak <- function(gene_name) {
    TIMP1 <- as.data.frame(PRO@assays$RNA@data[which(rownames(PRO@assays$RNA@data) == gene_name), ])
    colnames(TIMP1) <- "Expression"
    TIMP1$barcode <- rownames(TIMP1)
    PRO@meta.data$barcode <- rownames(PRO@meta.data)
    TIMP_dat <- merge(PRO@meta.data, TIMP1, by = "barcode")
    rownames(TIMP_dat) <- TIMP_dat$barcode
    TIMP_dat <- TIMP_dat[, -1]
    
    allcolour <- c("#0E98B2", "#F93F40")
    p <- ggplot(TIMP_dat, aes(x = braak, y = Expression, fill = diagnosis)) +
        ggtitle(gene_name) +
        geom_violin(trim = TRUE, scale = "width") +
        geom_signif(comparisons = my_comparisons, map_signif_level = FALSE, test = wilcox.test,
                    y_position = max(TIMP_dat$Expression) - 0.5, textsize = 6) +
        stat_summary(aes(label = round(after_stat(y), 2)), fun = "mean", geom = "text", vjust = -1.5) +
        scale_fill_manual(values = allcolour) +
        theme_bw() +
        theme(panel.grid.major = element_blank(),
              panel.grid.minor = element_blank(),
              axis.text.x = element_text(angle = 45, hjust = 1)) +
        labs(y = "Expression Level")
    print(p)
}

# Function to plot gene expression violin plot by age
do_genevln_age <- function(gene_name) {
    TIMP1 <- as.data.frame(PRO@assays$RNA@data[which(rownames(PRO@assays$RNA@data) == gene_name), ])
    colnames(TIMP1) <- "Expression"
    TIMP1$barcode <- rownames(TIMP1)
    PRO@meta.data$barcode <- rownames(PRO@meta.data)
    TIMP_dat <- merge(PRO@meta.data, TIMP1, by = "barcode")
    rownames(TIMP_dat) <- TIMP_dat$barcode
    TIMP_dat <- TIMP_dat[, -1]
    
    allcolour <- c("#0E98B2", "#F93F40")
    p <- ggplot(TIMP_dat, aes(x = age, y = Expression, fill = diagnosis)) +
        ggtitle(gene_name) +
        geom_violin(trim = TRUE, scale = "width") +
        geom_signif(comparisons = my_comparisons, map_signif_level = FALSE, test = wilcox.test,
                    y_position = max(TIMP_dat$Expression) - 0.5, textsize = 6) +
        stat_summary(aes(label = round(after_stat(y), 2)), fun = "mean", geom = "text", vjust = -1.5) +
        scale_fill_manual(values = allcolour) +
        theme_bw() +
        theme(panel.grid.major = element_blank(),
              panel.grid.minor = element_blank(),
              axis.text.x = element_text(angle = 45, hjust = 1)) +
        labs(y = "Expression Level")
    print(p)
}

# Set cell type identity
Idents(PRO) <- "celltype_global"

# Visualize gene expression for CLU
genenames <- "CLU"
options(repr.plot.height = 5, repr.plot.width = 12)
FeaturePlot(PRO, features = genenames, split.by = "diagnosis", raster = FALSE, label = TRUE)
options(repr.plot.height = 5, repr.plot.width = 6)
do_genevln(genenames)
do_genevln_braak(genenames)
do_genevln_age(genenames)

# Optional: Display metadata summaries
# table(PRO@meta.data$braak)
# table(PRO@meta.data$age)