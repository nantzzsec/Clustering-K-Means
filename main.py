# Clustering K-Means
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Membaca file CSV
file_path = "data.csv"  # Ganti dengan path file CSV Anda
data = pd.read_csv(file_path)

# Menampilkan beberapa baris pertama data
print("Data awal:")
print(data.head())

# Seleksi kolom yang akan digunakan untuk clustering
selected_columns = ["Pendapatan", "Pengeluaran"]  # Ganti dengan kolom sesuai dataset Anda
data_for_clustering = data[selected_columns]

# Normalisasi data (standarisasi)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_for_clustering)

# Menentukan jumlah cluster optimal dengan metode Elbow
wcss = []
range_clusters = range(1, 11)  # Mengevaluasi 1 hingga 10 cluster
for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# Plot WCSS untuk menentukan "Elbow"
plt.plot(range_clusters, wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.show()

# Menjalankan K-Means dengan jumlah cluster optimal (misalnya, k=3)
optimal_k = 3  # Tentukan jumlah cluster optimal berdasarkan grafik elbow
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
data['Cluster'] = kmeans.fit_predict(scaled_data)

# Menyimpan hasil clustering ke file CSV baru
output_file = "hasil_clustering.csv"
data.to_csv(output_file, index=False)
print(f"Hasil clustering disimpan di: {output_file}")

# Visualisasi hasil clustering (untuk 2D data)
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=data['Cluster'], cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='x', label='Centroid')
plt.title('Hasil Clustering dengan K-Means')
plt.xlabel(selected_columns[0])
plt.ylabel(selected_columns[1])
plt.legend()
plt.show()
