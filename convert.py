from stl import mesh

# 讀取 STL 檔案
your_mesh = mesh.Mesh.from_file(r"C:\Users\vivian\OneDrive - NTHU\文件\NTHU\NEAF\ITRI_project\grinding\Workpieces\工件轉接治具.stl")

# 定義縮放因子（例如放大 2 倍）
scale_factor = 0.0254  # 將單位從 mm 轉換為 m

# 對所有頂點進行縮放
your_mesh.vectors *= scale_factor

# 儲存為新的 STL 檔案
your_mesh.save(r"C:\Users\vivian\OneDrive - NTHU\文件\NTHU\NEAF\ITRI_project\grinding\Workpieces\simulation\Workpiece_transfer.stl")
