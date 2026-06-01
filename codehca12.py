import numpy as np
from mpi4py import MPI

N = 1000
TARGET = 5
NDIM = 12  # ganti angka ini saja kalau mau tambah dimensi lagi

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    np.random.seed(42)
    usia         = np.random.randint(15, 71, size=N)
    pendapatan   = np.random.randint(500000, 20000001, size=N)
    frekuensi    = np.random.randint(1, 11, size=N)
    waktu_kunjungan = np.random.randint(0, 3, size=N)       # 0=pagi,1=siang,2=malam
    durasi          = np.random.randint(60, 181, size=N)     # menit
    jumlah_tiket    = np.random.randint(1, 6, size=N)
    kepuasan        = np.random.randint(1, 6, size=N)        # skala 1-5
    membership      = np.random.randint(0, 2, size=N)        # 0=tidak,1=ya
    genre           = np.random.randint(0, 5, size=N)        # 5 genre
    snack           = np.random.randint(0, 6, size=N)
    kursi           = np.random.randint(1, 7, size=N)
    jenis_layar     = np.random.randint(0, 3, size=N)        # 0=reguler,1=premium,2=IMAX
    data_mentah = np.column_stack((usia, pendapatan, frekuensi,waktu_kunjungan, durasi, jumlah_tiket, 
                                   kepuasan, membership, genre, snack, kursi, jenis_layar)).astype(np.int32)
else:
    data_mentah = None

counts_rows = np.array([N // size + (1 if r < (N % size) else 0) for r in range(size)], dtype=np.int32)
displs_rows = np.array([sum(counts_rows[:r]) for r in range(size)], dtype=np.int32)

counts_elem = counts_rows * NDIM
displs_elem = displs_rows * NDIM

local_n    = counts_rows[rank]
local_data = np.empty((local_n, NDIM), dtype=np.int32)

comm.Scatterv([data_mentah, counts_elem, displs_elem, MPI.INT], local_data)

local_cat = np.zeros_like(local_data)
for i in range(local_n):
    umur, pend, freq, waktu, dur, tiket, kep, memb, gen, snk, krs, layar = local_data[i]
    if umur < 25:    local_cat[i, 0] = 0
    elif umur < 45:  local_cat[i, 0] = 1
    else:            local_cat[i, 0] = 2
    local_cat[i, 1] = 0 if pend < 5000000 else 1
    local_cat[i, 2] = 0 if freq < 4 else 1
    local_cat[i, 3] = waktu                              # sudah 0/1/2
    local_cat[i, 4] = 0 if dur < 90 else 1 if dur < 150 else 2
    local_cat[i, 5] = 0 if tiket <= 2 else 1
    local_cat[i, 6] = 0 if kep < 3 else 1
    local_cat[i, 7] = memb                               # sudah 0/1
    local_cat[i, 8] = gen                                # sudah 0-4
    local_cat[i, 9] = 0 if snk < 2 else 1
    local_cat[i,10] = 0 if krs <= 2 else 1
    local_cat[i,11] = layar                              # sudah 0/1/2

if rank == 0:
    data_full = np.empty((N, NDIM), dtype=np.int32)
else:
    data_full = None

comm.Gatherv(local_cat, [data_full, counts_elem, displs_elem, MPI.INT])
data_full = comm.bcast(data_full, root=0)

# --- 2. MEMBANGUN MATRIKS JARAK (tidak ada perubahan) ---
start_i = (rank * N) // size
end_i   = ((rank + 1) * N) // size if rank != size - 1 else N

local_entries  = []
local_min_dist = np.inf
local_min_i    = -1
local_min_j    = -1

for i in range(start_i, end_i):
    for j in range(i + 1, N):
        diff = data_full[i].astype(float) - data_full[j].astype(float)
        dist = np.sqrt(np.sum(diff**2))
        local_entries.append((i, j, dist))
        if dist < local_min_dist:
            local_min_dist = dist
            local_min_i    = i
            local_min_j    = j

local_pack = np.array([local_min_dist, local_min_i, local_min_j])
if rank == 0:
    all_mins = np.zeros((size, 3), dtype=float)
else:
    all_mins = None
comm.Gather(local_pack, all_mins, root=0)

n_local    = len(local_entries)
all_nlocal = comm.gather(n_local, root=0)

if rank == 0:
    total_entries = sum(all_nlocal)
    all_i = np.empty(total_entries, dtype=np.int32)
    all_j = np.empty(total_entries, dtype=np.int32)
    all_d = np.empty(total_entries, dtype=np.float64)
    off   = np.zeros(size, dtype=int)
    for r in range(1, size):
        off[r] = off[r - 1] + all_nlocal[r - 1]
else:
    all_i = None; all_j = None; all_d = None
    off   = None

if local_entries:
    local_i = np.array([e[0] for e in local_entries], dtype=np.int32)
    local_j = np.array([e[1] for e in local_entries], dtype=np.int32)
    local_d = np.array([e[2] for e in local_entries], dtype=np.float64)
else:
    local_i = np.empty(0, dtype=np.int32)
    local_j = np.empty(0, dtype=np.int32)
    local_d = np.empty(0, dtype=np.float64)

comm.Gatherv(local_i, [all_i, all_nlocal, off, MPI.INT])
comm.Gatherv(local_j, [all_j, all_nlocal, off, MPI.INT])
comm.Gatherv(local_d, [all_d, all_nlocal, off, MPI.DOUBLE])

if rank == 0:
    dist_matrix = np.zeros((N, N))
    for i, j, d in zip(all_i, all_j, all_d):
        dist_matrix[i, j] = d
        dist_matrix[j, i] = d

    best            = np.argmin(all_mins[:, 0])
    global_min_dist = all_mins[best, 0]
    global_min_i    = int(all_mins[best, 1])
    global_min_j    = int(all_mins[best, 2])

    dist_matrix  = comm.bcast(dist_matrix,  root=0)
    global_min_i = comm.bcast(global_min_i, root=0)
    global_min_j = comm.bcast(global_min_j, root=0)
else:
    dist_matrix  = comm.bcast(None, root=0)
    global_min_i = comm.bcast(None, root=0)
    global_min_j = comm.bcast(None, root=0)

# --- 3. INISIALISASI KLASTER ---
active       = np.ones(N, dtype=bool)
cluster_size = np.ones(N, dtype=int)
num_clusters = N
if rank == 0:
    members    = {i: [i] for i in range(N)}
    start_time = MPI.Wtime()
else:
    members    = None
    start_time = None

active       = comm.bcast(active,       root=0)
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
    size_new   = size_i_old + size_j_old

    if rank == 0:
        cluster_size[merge_i] = size_new
        members[merge_i].extend(members[merge_j])
        del members[merge_j]
        active[merge_j]  = False
        num_clusters    -= 1

    active       = comm.bcast(active,       root=0)
    cluster_size = comm.bcast(cluster_size, root=0)
    num_clusters = comm.bcast(num_clusters, root=0)
    size_i_old   = comm.bcast(size_i_old,   root=0)
    size_j_old   = comm.bcast(size_j_old,   root=0)
    size_new     = comm.bcast(size_new,     root=0)

    local_c_updates = []
    local_d_updates = []
    for c in range(rank * N // size, (rank + 1) * N // size if rank != size - 1 else N):
        if not active[c] or c == merge_i:
            continue
        d_ic  = dist_matrix[merge_i, c]
        d_jc  = dist_matrix[merge_j, c]
        new_d = (size_i_old * d_ic + size_j_old * d_jc) / size_new
        local_c_updates.append(c)
        local_d_updates.append(new_d)

    n_local_updates = len(local_c_updates)
    all_nupdates    = comm.gather(n_local_updates, root=0)

    if rank == 0:
        all_nupdates_arr = np.array(all_nupdates, dtype=np.int32)
        total_updates    = sum(all_nupdates)
        all_c            = np.empty(total_updates, dtype=np.int32)
        all_d            = np.empty(total_updates, dtype=np.float64)
        offset_upd       = np.zeros(size, dtype=np.int32)
        for r in range(1, size):
            offset_upd[r] = offset_upd[r - 1] + all_nupdates_arr[r - 1]
    else:
        all_c = None; all_d = None
        offset_upd = None

    local_c_arr = np.array(local_c_updates, dtype=np.int32)
    local_d_arr = np.array(local_d_updates, dtype=np.float64)

    comm.Gatherv(local_c_arr, [all_c, all_nupdates, offset_upd, MPI.INT])
    comm.Gatherv(local_d_arr, [all_d, all_nupdates, offset_upd, MPI.DOUBLE])

    if rank == 0:
        for c, d in zip(all_c, all_d):
            dist_matrix[merge_i, c] = d
            dist_matrix[c, merge_i] = d
        dist_matrix[merge_j, :] = np.inf
        dist_matrix[:, merge_j] = np.inf
    dist_matrix = comm.bcast(dist_matrix, root=0)

    local_min_d = np.inf
    local_min_i = -1
    local_min_j = -1
    for i in range(start_i, end_i):
        if not active[i]:
            continue
        for j in range(i + 1, N):
            if not active[j]:
                continue
            if dist_matrix[i, j] < local_min_d:
                local_min_d = dist_matrix[i, j]
                local_min_i = i
                local_min_j = j

    local_pack = np.array([local_min_d, local_min_i, local_min_j])
    all_mins   = None
    if rank == 0:
        all_mins = np.zeros((size, 3))
    comm.Gather(local_pack, all_mins, root=0)

    if rank == 0:
        best            = np.argmin(all_mins[:, 0])
        global_min_dist = all_mins[best, 0]
        global_min_i    = int(all_mins[best, 1])
        global_min_j    = int(all_mins[best, 2])
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
            anggota    = members[c]
            usia_mean  = np.mean(data_mentah[anggota, 0])
            pend_mean  = np.mean(data_mentah[anggota, 1])
            freq_mean  = np.mean(data_mentah[anggota, 2])
            waktu_mean  = np.mean(data_mentah[anggota, 3])
            durasi_mean = np.mean(data_mentah[anggota, 4])
            tiket_mean  = np.mean(data_mentah[anggota, 5])
            kep_mean    = np.mean(data_mentah[anggota, 6])
            memb_mean   = np.mean(data_mentah[anggota, 7])
            genre_mean  = np.mean(data_mentah[anggota, 8])
            snack_mean  = np.mean(data_mentah[anggota, 9])
            kursi_mean  = np.mean(data_mentah[anggota,10])
            layar_mean  = np.mean(data_mentah[anggota,11])
            print(f"  Klaster {c}: {len(anggota)} penonton | "
                f"rata2 usia {usia_mean:.1f}, pendapatan {pend_mean:.0f}, "
                f"frekuensi {freq_mean:.1f}, waktu {waktu_mean:.1f}, "
                f"durasi {durasi_mean:.1f}, tiket {tiket_mean:.1f}, "
                f"kepuasan {kep_mean:.1f}, membership {memb_mean:.1f}, "
                f"genre {genre_mean:.1f}, snack {snack_mean:.1f}, "
                f"kursi {kursi_mean:.1f}, layar {layar_mean:.1f}")
            
#cd "d:/semester 4/PemPar/tugas/projek uts"
#mpiexec -n 10 "C:\Users\ACER\AppData\Local\Python\pythoncore-3.14-64\python.exe" codehca12.py  