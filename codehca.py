import numpy as np
from mpi4py import MPI
import time

N = 1000               
TARGET = 5

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    np.random.seed(42)
    usia = np.random.randint(15, 71, size=N)
    pendapatan = np.random.randint(500000, 20000001, size=N)
    frekuensi = np.random.randint(1, 11, size=N)
    data_mentah = np.column_stack((usia, pendapatan, frekuensi)).astype(np.int32)
else:
    data_mentah = None

# Hitung jumlah BARIS per proses
counts_rows = np.array([N // size + (1 if r < (N % size) else 0) for r in range(size)], dtype=np.int32)
displs_rows = np.array([sum(counts_rows[:r]) for r in range(size)], dtype=np.int32)

# Ubah menjadi jumlah ELEMEN (karena setiap baris memiliki 3 fitur)
counts_elem = counts_rows * 3
displs_elem = displs_rows * 3

local_n = counts_rows[rank]
local_data = np.empty((local_n, 3), dtype=np.int32)

# Scatter dengan counts elemen
comm.Scatterv([data_mentah, counts_elem, displs_elem, MPI.INT], local_data)

# Kategorisasi lokal
local_cat = np.zeros_like(local_data)
for i in range(local_n):
    umur, pend, freq = local_data[i]
    # Usia
    if umur < 25:      local_cat[i,0] = 0
    elif umur < 45:    local_cat[i,0] = 1
    else:              local_cat[i,0] = 2
    # Pendapatan
    local_cat[i,1] = 0 if pend < 5000000 else 1
    # Frekuensi
    local_cat[i,2] = 0 if freq < 4 else 1

# Gather hasil kategorisasi (dengan counts elemen)
if rank == 0:
    data_full = np.empty((N, 3), dtype=np.int32)
else:
    data_full = None

comm.Gatherv(local_cat, [data_full, counts_elem, displs_elem, MPI.INT])
data_full = comm.bcast(data_full, root=0)

# --- 2. MEMBANGUN MATRIKS JARAK AWAL SECARA PARALEL ---
start_i = (rank * N) // size
end_i = ((rank + 1) * N) // size if rank != size-1 else N

local_entries = []
local_min_dist = np.inf
local_min_i = -1
local_min_j = -1

for i in range(start_i, end_i):
    for j in range(i+1, N):
        diff = data_full[i].astype(float) - data_full[j].astype(float)
        dist = np.sqrt(np.sum(diff**2))
        local_entries.append((i, j, dist))
        if dist < local_min_dist:
            local_min_dist = dist
            local_min_i = i
            local_min_j = j
            
# Kumpulkan minimum lokal dari setiap rank
local_pack = np.array([local_min_dist, local_min_i, local_min_j])
if rank == 0:
    all_mins = np.zeros((size, 3), dtype=float)
else:
    all_mins = None
comm.Gather(local_pack, all_mins, root=0)

# Kumpulkan entri jarak (PISAH i, j, dist)
n_local = len(local_entries)
all_nlocal = comm.gather(n_local, root=0)

if rank == 0:
    total_entries = sum(all_nlocal)
    # Siapkan array untuk menampung semua data
    all_i = np.empty(total_entries, dtype=np.int32)
    all_j = np.empty(total_entries, dtype=np.int32)
    all_d = np.empty(total_entries, dtype=np.float64)
    # Displacement untuk setiap rank
    off = np.zeros(size, dtype=int)
    for r in range(1, size):
        off[r] = off[r-1] + all_nlocal[r-1]
else:
    all_i = None; all_j = None; all_d = None
    off = None

# Siapkan data lokal sebagai array terpisah
if local_entries:
    local_i = np.array([e[0] for e in local_entries], dtype=np.int32)
    local_j = np.array([e[1] for e in local_entries], dtype=np.int32)
    local_d = np.array([e[2] for e in local_entries], dtype=np.float64)
else:
    local_i = np.empty(0, dtype=np.int32)
    local_j = np.empty(0, dtype=np.int32)
    local_d = np.empty(0, dtype=np.float64)

# Kirim masing-masing
comm.Gatherv(local_i, [all_i, all_nlocal, off, MPI.INT])
comm.Gatherv(local_j, [all_j, all_nlocal, off, MPI.INT])
comm.Gatherv(local_d, [all_d, all_nlocal, off, MPI.DOUBLE])

# Rank 0 menyusun matriks
if rank == 0:
    dist_matrix = np.zeros((N, N))
    for i, j, d in zip(all_i, all_j, all_d):
        dist_matrix[i, j] = d
        dist_matrix[j, i] = d

    # Minimum global dari minimum lokal (sudah dikumpulkan sebelumnya)
    best = np.argmin(all_mins[:, 0])
    global_min_dist = all_mins[best, 0]
    global_min_i = int(all_mins[best, 1])
    global_min_j = int(all_mins[best, 2])

    # Broadcast matriks dan minimum global
    dist_matrix = comm.bcast(dist_matrix, root=0)
    global_min_i = comm.bcast(global_min_i, root=0)
    global_min_j = comm.bcast(global_min_j, root=0)
else:
    dist_matrix = comm.bcast(None, root=0)
    global_min_i = comm.bcast(None, root=0)
    global_min_j = comm.bcast(None, root=0)

# --- 3. INISIALISASI KLASTER ---
active = np.ones(N, dtype=bool)
cluster_size = np.ones(N, dtype=int)
num_clusters = N
if rank == 0:
    members = {i: [i] for i in range(N)}
    start_time = MPI.Wtime()
else:
    members = None
    start_time = None

# Broadcast status awal
active = comm.bcast(active, root=0)
cluster_size = comm.bcast(cluster_size, root=0)

# --- 4–6. LOOP PENGGABUNGAN ---
while num_clusters > TARGET:
    merge_i = global_min_i
    merge_j = global_min_j
    if merge_i == -1 or merge_j == -1:
        if rank == 0:
            print("Tidak ada pasangan valid. Loop berhenti.")
        break

    size_i_old = cluster_size[merge_i]
    size_j_old = cluster_size[merge_j]
    size_new = size_i_old + size_j_old

    if rank == 0:
        cluster_size[merge_i] = size_new
        members[merge_i].extend(members[merge_j])
        del members[merge_j]
        active[merge_j] = False
        num_clusters -= 1

    # Broadcast status baru
    active = comm.bcast(active, root=0)
    cluster_size = comm.bcast(cluster_size, root=0)
    num_clusters = comm.bcast(num_clusters, root=0)
    size_i_old = comm.bcast(size_i_old, root=0)
    size_j_old = comm.bcast(size_j_old, root=0)
    size_new = comm.bcast(size_new, root=0)

    # Update matriks jarak dengan Average Linkage (subset tiap rank)
    local_c_updates = []
    local_d_updates = []
    for c in range(rank * N // size, (rank + 1) * N // size if rank != size-1 else N):
        if not active[c] or c == merge_i:
            continue
        d_ic = dist_matrix[merge_i, c]
        d_jc = dist_matrix[merge_j, c]
        new_d = (size_i_old * d_ic + size_j_old * d_jc) / size_new
        local_c_updates.append(c)
        local_d_updates.append(new_d)

    n_local_updates = len(local_c_updates)
    all_nupdates = comm.gather(n_local_updates, root=0)

    if rank == 0:
        # Pastikan all_nupdates sebagai array int32 untuk offset
        all_nupdates_arr = np.array(all_nupdates, dtype=np.int32)
        total_updates = sum(all_nupdates)
        all_c = np.empty(total_updates, dtype=np.int32)
        all_d = np.empty(total_updates, dtype=np.float64)
        offset_upd = np.zeros(size, dtype=np.int32)
        for r in range(1, size):
            offset_upd[r] = offset_upd[r-1] + all_nupdates_arr[r-1]
    else:
        all_c = None; all_d = None
        offset_upd = None
        all_nupdates_arr = None  # tidak digunakan

    # Siapkan array lokal untuk dikirim
    local_c_arr = np.array(local_c_updates, dtype=np.int32)
    local_d_arr = np.array(local_d_updates, dtype=np.float64)

    # Kirim dua array terpisah
    comm.Gatherv(local_c_arr, [all_c, all_nupdates, offset_upd, MPI.INT])
    comm.Gatherv(local_d_arr, [all_d, all_nupdates, offset_upd, MPI.DOUBLE])

    if rank == 0:
        for c, d in zip(all_c, all_d):
            dist_matrix[merge_i, c] = d
            dist_matrix[c, merge_i] = d
        dist_matrix[merge_j, :] = np.inf
        dist_matrix[:, merge_j] = np.inf
    dist_matrix = comm.bcast(dist_matrix, root=0)

    # Cari minimum lokal untuk iterasi berikutnya
    local_min_d = np.inf
    local_min_i = -1
    local_min_j = -1
    for i in range(start_i, end_i):
        if not active[i]:
            continue
        for j in range(i+1, N):
            if not active[j]:
                continue
            if dist_matrix[i, j] < local_min_d:
                local_min_d = dist_matrix[i, j]
                local_min_i = i
                local_min_j = j

    local_pack = np.array([local_min_d, local_min_i, local_min_j])
    all_mins = None
    if rank == 0:
        all_mins = np.zeros((size, 3))
    comm.Gather(local_pack, all_mins, root=0)

    if rank == 0:
        best = np.argmin(all_mins[:, 0])
        global_min_dist = all_mins[best, 0]
        global_min_i = int(all_mins[best, 1])
        global_min_j = int(all_mins[best, 2])
        global_min_i = comm.bcast(global_min_i, root=0)
        global_min_j = comm.bcast(global_min_j, root=0)
    else:
        global_min_i = comm.bcast(None, root=0)
        global_min_j = comm.bcast(None, root=0)

# --- 7. HASIL AKHIR ---
if rank == 0:
    elapsed = MPI.Wtime() - start_time
    print(f"\nTotal waktu eksekusi: {elapsed:.4f} detik")
    print("Profil klaster akhir:")
    for c in range(N):
        if active[c]:
            anggota = members[c]
            usia_mean = np.mean(data_mentah[anggota, 0])
            pend_mean = np.mean(data_mentah[anggota, 1])
            freq_mean = np.mean(data_mentah[anggota, 2])
            print(f"  Klaster {c}: {len(anggota)} penonton | "
                  f"rata2 usia {usia_mean:.1f}, pendapatan {pend_mean:.0f}, "
                  f"frekuensi {freq_mean:.1f}")
            
#cd "d:/semester 4/PemPar/tugas/projek uts"
#mpiexec -n 4 "C:\Users\ACER\AppData\Local\Python\pythoncore-3.14-64\python.exe" codehca.py            