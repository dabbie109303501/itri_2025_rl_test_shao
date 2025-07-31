from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IGESControl import IGESControl_Reader
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRep import BRep_Tool

# 載入 .igs 檔
reader = IGESControl_Reader()
f="J6RB_COVER_new"
status = reader.ReadFile("igs/"+f+".igs")

if status == IFSelect_RetDone:
    reader.TransferRoots()
    shape = reader.OneShape()

    # 做三角化 (mesh)
    mesh = BRepMesh_IncrementalMesh(shape, 0.1)  # 0.1 為網格精度，可調
    mesh.Perform()

    # 匯出成 STL
    stl_writer = StlAPI_Writer()
    stl_writer.Write(shape, "stl/"+f+".stl")

    print("轉換成功")
else:
    print("讀取 IGES 失敗")

# ------------------------------------------------

# from stl import mesh
# import trimesh

# # 讀取 STL 檔案
# your_mesh = mesh.Mesh.from_file("stl/"+f+".stl")

# # 定義縮放因子（例如放大 2 倍）
# scale_factor = 0.001  # 將單位從 mm 轉換為 m

# # 對所有頂點進行縮放
# your_mesh.vectors *= scale_factor

# # 儲存為新的 STL 檔案
# your_mesh.save("stl/"+f+"_0.001.stl")


