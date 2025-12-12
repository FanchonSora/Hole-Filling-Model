import kagglehub

# Download latest version
path = kagglehub.dataset_download("hajareddagni/shapenetcorev2")

print("Path to dataset files:", path)