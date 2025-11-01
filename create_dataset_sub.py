from data_scripts.Mesh_dataset import MeshDataset

dataset_path = "Data/meshData_org.npz"
save_dataset_path = "Data/meshData_org_onlychair.npz"
dataset = MeshDataset.load(dataset_path)

def filter_dataset_by_categories(dataset, categories_to_ft):
    new_data = [d for d in dataset.data if d['texts'] in categories_to_ft]
    return MeshDataset(new_data)

# Example:
categories_to_ft = ['chair']
ft_dataset = filter_dataset_by_categories(dataset, categories_to_ft)

labels = list(set(item["texts"] for item in ft_dataset.data))
print(ft_dataset.data[0].keys())
print(labels)
print(len(ft_dataset.data))

# Select 5 unique chair samples
unique_chairs = ft_dataset.data[:5]

# Duplicate each sample 500 times
duplicated_chairs = []
for chair in unique_chairs:
    duplicated_chairs.extend([chair] * 500)

# Update dataset
ft_dataset.data = duplicated_chairs
print("Total chairs after duplication:", len(ft_dataset.data))  # Should be 2500

ft_dataset.generate_face_edges()
ft_dataset.save(save_dataset_path)