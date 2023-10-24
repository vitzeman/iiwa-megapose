import open3d as o3d


cleared = "/home/vit/Documents/DP/3Dreconstuct/meshes_scaled_cleared/d01_controller"
orig = "/home/vit/Documents/DP/3Dreconstuct/meshes_scaled/d01_controller"


cleared_mesh = o3d.io.read_triangle_mesh(
    cleared + "/mesh.obj", True
)  # True needs to be set to read texture

orig_mesh = o3d.io.read_triangle_mesh(
    orig + "/mesh.obj", True
)  # True needs to be set to read texture


o3d.visualization.draw_geometries([cleared_mesh, orig_mesh])
