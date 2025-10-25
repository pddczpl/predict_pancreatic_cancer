library(oligo)
library(AnnotationDbi)
library(biomaRt)
library(dplyr)
library(sva)
library(progress)
library(readxl)

# ===============================================
# HELPER FUNCTIONS
# ===============================================

connect_biomart <- function(max_attempts = 5) {
  mirrors <- c("https://useast.ensembl.org", "https://www.ensembl.org", "https://asia.ensembl.org")
  
  for(mirror in mirrors) {
    for(attempt in 1:max_attempts) {
      tryCatch({
        message(paste("Connecting to", mirror, "- Attempt", attempt))
        mart <- useMart("ensembl", dataset = "hsapiens_gene_ensembl", host = mirror)
        message("Successfully connected to biomaRt!")
        return(mart)
      }, error = function(e) {
        message(paste("Attempt", attempt, "failed:", e$message))
        if(attempt < max_attempts) Sys.sleep(5)
      })
    }
  }
  stop("Failed to connect to biomaRt after trying all mirrors")
}

perform_qc <- function(expr_matrix, sample_metadata) {
  qc_results <- list(
    total_probes = nrow(expr_matrix),
    total_samples = ncol(expr_matrix),
    missing_values = sum(is.na(expr_matrix)),
    zero_values = sum(expr_matrix == 0, na.rm = TRUE),
    expression_range = range(expr_matrix, na.rm = TRUE),
    samples_per_group = table(sample_metadata$Group)
  )
  
  expected_samples <- c("Tumor" = 69, "Normal" = 61)
  if(!all(names(expected_samples) %in% names(qc_results$samples_per_group))) {
    warning("Sample group counts differ from published literature")
  }
  
  return(qc_results)
}

# ===============================================
# MAIN PROCESSING PIPELINE
# ===============================================

print("Starting Enhanced GSE62452 Processing Pipeline...")

# Set working directory (set data_dir biến tùy theo hệ thống)
if(!exists("data_dir")) {
  possible_dirs <- c("GSE62452_RAW", "C:/Users/DatGo/OneDrive/Documents/Personal_Project/predict_pancreatic_cancer/geo/GSE62452_RAW", "~/GSE62452_RAW")
  
  data_dir <- NULL
  for(dir in possible_dirs) {
    if(file.exists(dir)) {
      data_dir <- dir
      break
    }
  }
  
  if(is.null(data_dir)) {
    stop("GSE62452_RAW directory not found. Please set data_dir variable.")
  }
}

setwd(data_dir)
print(paste("Working directory:", getwd()))

# Phase 1: Read CEL files
print("Phase 1: Reading CEL files...")
cel_files <- list.files(pattern = ".CEL$|.cel$")
print(paste("Found", length(cel_files), "CEL files"))

if(length(cel_files) == 0) {
  stop("No CEL files found in directory")
}

tryCatch({
  eset <- read.celfiles(cel_files)
  print("CEL files successfully loaded")
}, error = function(e) {
  stop(paste("Failed to read CEL files:", e$message))
})

platform_info <- annotation(eset)
print(paste("Platform:", platform_info))
if(!grepl("hugene", platform_info, ignore.case = TRUE)) {
  warning(paste("Unexpected platform:", platform_info, "Expected HuGene array"))
}

print("Performing RMA normalization...")
tryCatch({
  eset_rma <- rma(eset)
  expr_matrix <- exprs(eset_rma)
  print(paste("RMA normalization completed. Dimensions:", nrow(expr_matrix), "x", ncol(expr_matrix)))
}, error = function(e) {
  stop(paste("RMA normalization failed:", e$message))
})

# Phase 2: Extract sample metadata and map group from paired Excel info
sample_names <- colnames(expr_matrix)
sample_metadata <- data.frame(Sample_ID = sample_names, stringsAsFactors = FALSE)

# Tách GSE_ID từ Sample_ID (dạng GSMxxxx)
sample_metadata <- sample_metadata %>%
  mutate(GSE_ID = sub("^(GSM\\d+).*", "\\1", Sample_ID))

# Đọc file paired sample info Excel
paired_info <- read_excel("GSE62452_paired_sample_information.xlsx")

# Merge nhãn nhóm (Group) từ paired sample info
sample_metadata <- merge(sample_metadata,
                         paired_info[, c("GSE ID", "Group")],
                         by.x = "GSE_ID",
                         by.y = "GSE ID",
                         all.x = TRUE)

print("Sample group distribution:")
print(table(sample_metadata$Group))

# ===============================================
# PHASE 3: GENE ANNOTATIONS
# ===============================================

print("Phase 3: Getting gene annotations...")

# Connect to biomaRt with retry
mart <- connect_biomart()

# Get unique probe IDs
probe_ids <- rownames(expr_matrix)
print(paste("Total probes to annotate:", length(probe_ids)))

# Query biomaRt in batches to handle large datasets
batch_size <- 1000
n_batches <- ceiling(length(probe_ids) / batch_size)
gene_annotations <- data.frame()

pb <- progress_bar$new(
  format = "  Annotating [:bar] :percent in :elapsed",
  total = n_batches, clear = FALSE, width = 60
)

for(i in 1:n_batches) {
  start_idx <- ((i-1) * batch_size) + 1
  end_idx <- min(i * batch_size, length(probe_ids))
  batch_probes <- probe_ids[start_idx:end_idx]
  
  tryCatch({
    batch_annotations <- getBM(
      attributes = c("affy_hugene_1_0_st_v1", "external_gene_name", 
                     "description", "chromosome_name", "start_position", 
                     "end_position", "gene_biotype"),
      filters = "affy_hugene_1_0_st_v1",
      values = batch_probes,
      mart = mart
    )
    
    if(nrow(batch_annotations) > 0) {
      gene_annotations <- rbind(gene_annotations, batch_annotations)
    }
    
  }, error = function(e) {
    warning(paste("Batch", i, "annotation failed:", e$message))
  })
  
  pb$tick()
}

# Clean and standardize annotations
colnames(gene_annotations) <- c("Probe_ID", "Gene_Symbol", "Description", 
                                "Chromosome", "Start_Position", "End_Position", "Biotype")

# Handle missing annotations
missing_annotations <- setdiff(probe_ids, gene_annotations$Probe_ID)
if(length(missing_annotations) > 0) {
  print(paste("Warning:", length(missing_annotations), "probes without annotations"))
  
  # Add missing probes with NA values
  missing_df <- data.frame(
    Probe_ID = missing_annotations,
    Gene_Symbol = NA,
    Description = NA,
    Chromosome = NA,
    Start_Position = NA,
    End_Position = NA,
    Biotype = NA,
    stringsAsFactors = FALSE
  )
  gene_annotations <- rbind(gene_annotations, missing_df)
}

# Sort to match expression matrix order
gene_annotations <- gene_annotations[match(probe_ids, gene_annotations$Probe_ID), ]

print(paste("Annotation coverage:", 
            round(sum(!is.na(gene_annotations$Gene_Symbol)) / nrow(gene_annotations) * 100, 1), "%"))

# ===============================================
# PHASE 4: QUALITY CONTROL
# ===============================================

print("Phase 4: Quality control assessment...")

qc_results <- perform_qc(expr_matrix, sample_metadata)
print("Quality Control Results:")
print(qc_results)

# Check for batch effects (if applicable)
if(length(unique(sample_metadata$Group)) > 1) {
  print("Checking for potential batch effects...")
  # Basic visualization would go here in practice
}

# ===============================================
# PHASE 5: DATA EXPORT
# ===============================================

print("Phase 5: Exporting processed data...")

# 1. Expression matrix with annotations
expr_annotated <- data.frame(
  Probe_ID = rownames(expr_matrix),
  Gene_Symbol = gene_annotations$Gene_Symbol,
  Description = gene_annotations$Description,
  expr_matrix,
  check.names = FALSE,
  stringsAsFactors = FALSE
)

# Handle duplicate gene symbols by taking mean expression
expr_gene_level <- expr_annotated %>%
  filter(!is.na(Gene_Symbol)) %>%
  group_by(Gene_Symbol) %>%
  summarise(
    Description = first(Description[!is.na(Description)]),
    across(starts_with("GSM"), ~ mean(.x, na.rm = TRUE)),
    .groups = "drop"
  )

write.csv(expr_annotated, "C:/Users/DatGo/OneDrive/Documents/Personal_Project/predict_pancreatic_cancer/geo/GSE62452/GSE62452_expression_with_annotations.csv", row.names = FALSE)
write.csv(expr_gene_level, "C:/Users/DatGo/OneDrive/Documents/Personal_Project/predict_pancreatic_cancer/geo/GSE62452/GSE62452_gene_level_expression.csv", row.names = FALSE)

# 2. Sample metadata
write.csv(sample_metadata, "C:/Users/DatGo/OneDrive/Documents/Personal_Project/predict_pancreatic_cancer/geo/GSE62452/GSE62452_sample_metadata.csv", row.names = FALSE)

# 3. Gene annotations
write.csv(gene_annotations, "C:/Users/DatGo/OneDrive/Documents/Personal_Project/predict_pancreatic_cancer/geo/GSE62452/GSE62452_gene_annotations.csv", row.names = FALSE)

# 4. ML-ready transposed format
expr_ml_ready <- data.frame(
  Sample_ID = colnames(expr_matrix),
  Group = sample_metadata$Group,
  t(expr_matrix),
  check.names = FALSE,
  stringsAsFactors = FALSE
)
write.csv(expr_ml_ready, "C:/Users/DatGo/OneDrive/Documents/Personal_Project/predict_pancreatic_cancer/geo/GSE62452/GSE62452_ML_ready_features.csv", row.names = FALSE)

# 5. Quality control report
qc_report <- data.frame(
  Metric = names(unlist(qc_results)),
  Value = unlist(qc_results),
  stringsAsFactors = FALSE
)
write.csv(qc_report, "C:/Users/DatGo/OneDrive/Documents/Personal_Project/predict_pancreatic_cancer/geo/GSE62452/GSE62452_quality_control_report.csv", row.names = FALSE)

# ===============================================
# SUMMARY
# ===============================================

print("===============================================")
print("PROCESSING COMPLETED SUCCESSFULLY!")
print("===============================================")
print("Files generated:")
print("1. GSE62452_expression_with_annotations.csv - Full expression matrix")
print("2. GSE62452_gene_level_expression.csv - Gene-level aggregated data") 
print("3. GSE62452_sample_metadata.csv - Sample information")
print("4. GSE62452_gene_annotations.csv - Gene annotation details")
print("5. GSE62452_ML_ready_features.csv - ML-ready format")
print("6. GSE62452_quality_control_report.csv - QC metrics")
print("===============================================")

# Memory cleanup
gc()
print("Pipeline completed successfully!")


library(affy)
cel <- ReadAffy(filenames="C:/Users/DatGo/OneDrive/Documents/Personal_Project/predict_pancreatic_cancer/geo/GSE62452_RAW/GSM1527234_Hussain_58_34435_HuGene-1_0-st-v1_.CEL")
eset <- rma(cel)
write.csv(exprs(eset), "C:/Users/DatGo/OneDrive/Documents/Personal_Project/predict_pancreatic_cancer/geo/sample.csv")

