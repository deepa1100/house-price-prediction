import kagglehub

# Download latest version
path = kagglehub.dataset_download(
    "sanjay3454chauhan/bangluru-house-dataset"
)

print("Path to dataset files:", path)