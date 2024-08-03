# Load libraries
library(httr)
library(jsonlite)
library(data.table)

# Set root directory & create subdirectories
path <- "path/to/root/directory/with/eddmaps/occurrence/data/"
dir.create(file.path(path, "csvs"), showWarnings = FALSE)
dir.create(file.path(path, "images"), showWarnings = FALSE)

# Load eddmaps occurrences
eddmaps_occurrences <- fread(paste0(path, "mappings.csv"))
# Sort by ReviewDate
eddmaps_occurrences <- eddmaps_occurrences[order(ReviewDate)]

# Function to get JSON data from a list of occurrence IDs using the Bugwood API
get_json_data <- function(ids) {
  # Base URL
  base_url <- "https://sandbox.bugwoodcloud.org/v2/occurrence/"
  # Initialize an empty list to store results
  results <- list()
  # Loop through each ID
  for (id in ids) {
    # Construct the URL
    url <- paste0(base_url, id)
    # Make the GET request
    response <- GET(url)
    # Check if the request was successful
    if (status_code(response) == 200) {
      # Capture the JSON content as a list
      content <- content(response, "text")
      data <- fromJSON(content)
      # Store the raw JSON data in the results list
      results[[as.character(id)]] <- data
    } else {
      # Handle failed request by storing NA
      results[[as.character(id)]] <- NA
    }
  }
  return(results)
}

# Function to unnest JSON data into a data table
unnest_json_data <- function(json_data_list) {
  # Unnest the JSON data into a data table
  json_data_table <- rbindlist(lapply(json_data_list, as.data.table),
    fill = TRUE
  )
  # Find columns that are lists
  list_cols <- sapply(json_data_table, is.list)
  list_cols <- names(json_data_table)[list_cols]
  # Flatten the list columns
  json_data_table[, (list_cols) := lapply(.SD, unlist), .SDcols = list_cols]
  return(json_data_table)
}

# Get JSON data for all occurrence IDs
occurrence_ids <- eddmaps_occurrences$objectid

# Split the occurrence IDs into chunks of 100 for efficient API calls
occurrence_chunks <- split(occurrence_ids,
  ceiling(seq_along(occurrence_ids) / 50)
)

# Loop through each chunk of occurrence IDs
for (i in seq_along(occurrence_chunks)) {
  # Get the JSON data for the chunk
  chunk_json_data <- get_json_data(occurrence_chunks[[i]])
  # Unnest the JSON data into a data table
  chunk_data <- unnest_json_data(chunk_json_data)
  retrieved <- length(unique(chunk_data$objectid))
  # Save the data to a CSV file
  fwrite(chunk_data,
    file = paste0(path, "csvs/json_dt_chunk_", i, ".csv"), row.names = FALSE
  )
  # Print progress
  message("Retrieved data for chunk ", i, " (", retrieved,
    " unique occurrences)"
  )
}

# Pull in csvs only the "objectid" and "images.imagename" columns
# Load the CSV files
json_files <- list.files(paste0(path, "csvs"),
  pattern = "json_dt_chunk_.*.csv", full.names = TRUE
)

# Read the CSV files into a list of data tables
json_data_list <- lapply(json_files, fread)

# Use lapply to change all column names to lower case and replace "." with "_"
json_data_list <- lapply(json_data_list, function(x) {
  setnames(x, tolower(names(x)))
  setnames(x, gsub("\\.", "_", names(x)))
  return(x)
})

# Use lapply to select only the desired columns: objectid and images_imagename
json_data_list <- lapply(json_data_list, function(x) {
  tryCatch(
    {x[, .(objectid, images_imagename)]
    },
    error = function(e) {
      # Handle the error when images_imagename is missing
      print("Error: images_imagename is missing")
      return(NULL)
    }
  )
})

# Rbindlist
json_data_wimage <- rbindlist(json_data_list, fill = TRUE)

# Remove images_imagenae NA
json_data_wimage <- json_data_wimage[images_imagename != ""]

# Base url for image download
base_url <- "https://secure.bugwoodcloud.org/eddmaps/images/report/768x512/"

# Function to paste0 base_url with image name, and then download
download_images <- function(data) {
  # Create a new column with the full image URL
  data[, image_url := paste0(base_url, images_imagename)] # nolint
  # Download the images
  for (i in 1:nrow(data)) { # nolint
  image_name <- data[i, images_imagename] # nolint
  image_url <- data[i, image_url]
  image_path <- file.path(paste0(path, "images"), image_name)
  # Check if the image already exists
  if (!file.exists(image_path)) {
    # Try to download the image with error handling
    tryCatch(
        {
          # Download the image
          GET(image_url, write_disk(image_path, overwrite = TRUE))
        },
        error = function(e) {
          # Print error message
          print(paste("Error downloading image:", image_name))
          # System sleep
          Sys.sleep(3)
        }
      )
    }
  }
}

# Split data into chunks of 100 for efficient image downloads
image_chunks <- split(json_data_wimage,
  ceiling(seq_along(json_data_wimage$objectid) / 50)
)

# Loop through each chunk of image data
for (i in seq_along(image_chunks)) {
  # Download the images for the chunk
  download_images(image_chunks[[i]])
  # Print progress
  message("Downloaded images for chunk ", i, "of", length(image_chunks))
}