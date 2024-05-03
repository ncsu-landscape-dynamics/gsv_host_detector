# Setup libraries & paths
library(data.table)
library(aws.s3)
path <- "C:/Users/talake2/Desktop/auto_arborist_cvpr2022_v015/pytorch_cnn_classifier/datasets/inat/" # change to your root folder for iNaturalist images

# Load autoarborist genera data
aa <- fread(paste0(path, "autoarborist/aa_sumstats.csv"))
# Drop unnecessary columns
names(aa)[names(aa) == "count"] <- "total_aa_imgs"

# 1. Download aws open data
inat_bucket <- "s3://inaturalist-open-data/"
item <- "metadata/inaturalist-open-data-latest.tar.gz"

# Helper to update the metadata based on monthly updates
if (!file.exists(paste0(path, item))) {
  save_object(
    object = item,
    bucket = inat_bucket,
    region = "us-east-1",
    file = paste0(path, item)
  )
  untar(paste0(path, "inaturalist-open-data-latest.tar.gz"), exdir = paste0(path, "metadata"))
} else {
  print("metadata exists; checking last modified date...")
  # If metadata already exists, check when it was last modified
  last_modified <- file.info(paste0(path, item))$mtime
  # If last modified was more than 31 days ago, download again
  if (as.numeric(Sys.time() - last_modified, units = "days") > 31) {
    save_object(
      object = item,
      bucket = inat_bucket,
      region = "us-east-1",
      file = paste0(path, item)
    )
    untar(paste0(path, "inaturalist-open-data-latest.tar.gz"), exdir = paste0(path, "metadata"))
  } else {
    print("metadata is up to date")
  }
}

# 2. Create subsets of inat data based rank, aa genera, quality & location
## Read in inaturalist metadata
files <- list.files(paste0(path, "metadata"), recursive = TRUE, full.names = TRUE)

taxa <- fread(files[grep("taxa.csv", files)])
observations <- fread(files[grep("observations.csv", files)])
photos <- fread(files[grep("photos.csv", files)])

# Keep rank == "species"; and split species into genus and species columns
taxa_sp <- taxa[rank == "species", ]
taxa_sp$name_split <- strsplit(taxa_sp$name, " ")
taxa_sp <- transform(taxa_sp,
  genus = sapply(name_split, `[`, 1),
  species = sapply(name_split, `[`, 2)
)
taxa_sp[, name_split := NULL]

# Keep rank == "genus" and copy name column to genus column
taxa_genus <- taxa[rank == "genus", ]
taxa_genus$genus <- taxa_genus$name

# Combine species and genus taxa
taxa <- rbindlist(list(taxa_sp, taxa_genus), fill = TRUE)

# Reformat genus column to lower and remove whitespace
taxa[, genus := tolower(genus)]
taxa[, genus := gsub(" ", "", genus)]

# Merge inaturalist taxa and autoarborist data
taxa <- merge(taxa, aa, by = "genus")

# Filter observations to only include research grade
observations <- observations[quality_grade == "research", ]

# Merge taxa with observations
observations <- merge(observations, taxa, by = "taxon_id")

# Merge observations with photos using observation_uuid column
observations <- merge(observations, photos, by = "observation_uuid")

# Remove unnecessary columns to save space & write out observations
observations[, rank_level := NULL]
observations[, observer_id.y := NULL]
observations[, ancestry := NULL]
observations[, quality_grade := NULL]
observations[, active := NULL]

# Write out global observations on aa genera
#fwrite(observations, paste0(path, "metadata/tree_observations_global.csv"),
#  row.names = FALSE
#)

# Filter observations to only include US/CANADA
observations_global <- observations
observations <- observations[latitude >= 24 & latitude <= 60 &
  longitude >= -135 & longitude <= -60, ]

# 3. Compute summary statistics for inat data globally and US/CANADA

# Calculate number of observations for each genus in observations
observations_counts <- unique(observations[, .N, by = genus])
names(observations_counts)[names(observations_counts) == "N"] <- "inat_imgs_USCAN"

observations_counts_global <- unique(observations_global[, .N, by = genus])
names(observations_counts_global)[names(observations_counts_global) == "N"] <- "inat_imgs_global"

# merge autoarborist with observations_counts based on genus
aa <- merge(aa, observations_counts, by = "genus", all.x = TRUE)
aa <- merge(aa, observations_counts_global, by = "genus", all.x = TRUE)

# Write out autoarborist
#fwrite(aa, paste0(path, "aa_inat_noimgs.csv"), row.names = FALSE)

# Subset genera for download

#small_set <- c(
#  "phoenix", "washingtonia", "pinus", "picea",
#  "juniperus", "thuja", "sequoia", "pseudotsuga", "taxodium", "acer",
#  "fraxinus", "quercus", "ulmus", "tilia", "pyrus", "gleditsia", "magnolia",
#  "betula", "ailanthus", "citrus", "juglans", "prunus", "rhus", "erythrina",
#  "cupaniopsis"
#)

# Use lapply to filter observations by small_set

subset_obs <- lapply(small_set, function(x) {
  observations[genus == x, ]
})
subset_obs <- rbindlist(subset_obs, fill = TRUE)

# Write out subset_obs
#fwrite(subset_obs, paste0(path, "metadata/subset_treeobs_uscan_autoarborist.csv"), row.names = FALSE)

# Create directories for images
subset_obs$folder_name <- paste0("/", subset_obs$genus, "/")
subset_obs[name == "Ailanthus altissima", ]$folder_name <- "/ailanthus_altissima/"
#subset_obs[name == "Juglans nigra", ]$folder_name <- "/juglans_nigra/"

classes <- unique(subset_obs$folder_name)

lapply(classes, function(x) {
  dir.create(paste0(path, "/images/original_full/", x),
    recursive = TRUE,
    showWarnings = FALSE
  )
})

# Append s3  filebucket name
subset_obs$s3filepath <- paste0(
  "photos/", subset_obs$photo_id, "/original.",
  subset_obs$extension
)
subset_obs$dlpath <- paste0(
  path, "images/original_full", subset_obs$folder_name,
  subset_obs$photo_id, ".",
  subset_obs$extension
)


# 4. Download aws open data

# Function to check if file exists
file_exists <- function(path) {
  fs::file_exists(path)
}

# 4. Download aws open data

# Filter and download the first 1000 images per class
lapply(classes, function(x) {
  imgs1k <- subset_obs[folder_name == x, ]
  inat_bucket <- "s3://inaturalist-open-data/"
  
  lapply(1:nrow(imgs1k), function(i) {
    item <- imgs1k$s3filepath[i]
    dlpath <- imgs1k$dlpath[i]
    
    # Check if file exists locally
    if (!file_exists(dlpath)) {
      tryCatch(
        expr = {
          save_object(
            object = item,
            bucket = inat_bucket,
            region = "us-east-1",
            file = dlpath
          )
        },
        error = function(e) {
          # Handle the error gracefully
          cat("Error downloading object:", item, "\n")
          cat("Error message:", conditionMessage(e), "\n")
        }
      )
    } else {
      cat("File already exists. Skipping download for:", dlpath, "\n")
    }
  })
})

# EOF

