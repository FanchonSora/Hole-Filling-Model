import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tkinter import Tk, Label, Button, Listbox, Scrollbar, filedialog, messagebox, END
import math
import tkinter as tk  

# ==================== ĐƯỜNG DẪN CHÍNH XÁC CHO BẠN ====================
BASE_DIR = r"C:\Users\nhat0\Downloads\Hole-Filling-Model\hole-filling"

DATA_FOLDER = os.path.join(BASE_DIR, "data", "ShapeNetCore.v2", "ShapeNetCore.v2")  
MODEL_PATH = os.path.join(BASE_DIR, "pcn_improved_v1_best(50).pth")  # Model của bạn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Đang dùng thiết bị: {device}")

# ==================== CẤU HÌNH ====================
CONFIG = {
    "num_partial_points": 2048,
    "num_complete_points": 2048,
    "feat_dim": 1024,
    "num_coarse": 2048,
    "num_fine": 16384,
}

# ==================== MÔ HÌNH PCN (giữ nguyên) ====================
class PointNetEncoder(nn.Module):
    def __init__(self, feat_dim=1024):
        super().__init__()
        # Deeper architecture: 3 → 64 → 128 → 256 → 512 → 1024
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)   
        self.conv4 = nn.Conv1d(256, 512, 1)   
        self.conv5 = nn.Conv1d(512, feat_dim, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)         
        self.bn4 = nn.BatchNorm1d(512)        
        self.bn5 = nn.BatchNorm1d(feat_dim)
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))    
        x = F.relu(self.bn4(self.conv4(x)))   
        x = self.bn5(self.conv5(x))
        x = torch.max(x, dim=2)[0]
        x = self.dropout(x)                     
        return x

class TransformerPointNetEncoder(nn.Module):
    def __init__(self, feat_dim=1024):
        super().__init__()
        
        # PointNet backbone
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        
        # Self-attention on point features
        self.self_attn = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Final projection
        self.conv5 = nn.Conv1d(512, feat_dim, 1)
        self.bn5 = nn.BatchNorm1d(feat_dim)
    
    def forward(self, x):
        # x: (B, N, 3)
        B, N, _ = x.shape
        
        x = x.transpose(2, 1)  # (B, 3, N)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))  # (B, 512, N)
        
        # Self-attention
        x_attn = x.transpose(2, 1)  # (B, N, 512)
        x_attn, _ = self.self_attn(x_attn, x_attn, x_attn)
        x = x + x_attn.transpose(2, 1)  # Residual connection
        
        # Global and local features
        feat_local = x.transpose(2, 1)  # (B, N, 512) - per-point features
        
        x = self.bn5(self.conv5(x))
        feat_global = torch.max(x, dim=2)[0]  # (B, feat_dim) - global feature
        
        return feat_global, feat_local

class CoarseDecoder(nn.Module):
    def __init__(self, feat_dim=1024, num_coarse=1024):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_coarse * 3)

        self.ln1 = nn.LayerNorm(1024)
        self.ln2 = nn.LayerNorm(1024)

        self.num_coarse = num_coarse

    def forward(self, feat):
        x = F.relu(self.ln1(self.fc1(feat)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        return x.view(-1, self.num_coarse, 3)

class FoldingRefinement(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 512, 1)
        self.gn1 = nn.GroupNorm(32, 512)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.gn2 = nn.GroupNorm(32, 512)
        self.conv3 = nn.Conv1d(512, 3, 1)

    def forward(self, x):
        # x: (B, C, P)
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.relu(self.gn2(self.conv2(x)))
        x = self.conv3(x)  # (B, 3, P)
        return x

class PCN(nn.Module):
    def __init__(self, feat_dim=1024, num_coarse=2048):
        super().__init__()

        self.encoder = TransformerPointNetEncoder(feat_dim)

        self.coarse_decoder = CoarseDecoder(feat_dim, num_coarse)

        # refine + folding
        self.folding1 = FoldingRefinement(feat_dim + 3 + 2)
        self.folding2 = FoldingRefinement(feat_dim + 3 + 2)
        self.folding3 = FoldingRefinement(feat_dim + 3 + 2)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # MLP refine coarse
        self.refine_mlp = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        # Convert coarse (B,N,3) → (B,N,feat_dim) để làm query
        self.coarse_feat_mlp = nn.Linear(3, feat_dim)

        # Convert local features 512 → 1024
        self.project_partial_feat = nn.Linear(512, feat_dim)
    def forward(self, partial):
        B = partial.shape[0]

        feat_global, feat_local = self.encoder(partial)
        feat_local = self.project_partial_feat(feat_local)   # (B, N, 1024)

        coarse = self.coarse_decoder(feat_global)            # (B, Nc, 3)

        coarse_ref = self.refine_with_attention(coarse, partial, feat_local)

        fine1 = self.expand_and_fold(feat_global, coarse_ref)
        fine2 = self.expand_and_fold(feat_global, fine1)
        fine3 = self.expand_and_fold(feat_global, fine2)

        return coarse, fine1, fine2, fine3
    def refine_with_attention(self, coarse, partial, partial_feat):
        B, Nc, _ = coarse.shape

        # Map (B,Nc,3) → (B,Nc,1024)
        coarse_feat = self.coarse_feat_mlp(coarse)

        refined_feat, _ = self.cross_attn(
            query=coarse_feat,
            key=partial_feat,
            value=partial_feat
        )

        delta = self.refine_mlp(refined_feat)
        return coarse + delta
    def expand_and_fold(self, feat_global, coarse_points):
        B, Nc, _ = coarse_points.shape

        # tạo grid 2D để folding
        grid = self.build_grid(B, Nc).to(feat_global.device)   # (B,2,Nc)

        # expand global feature
        feat_global_expand = feat_global.unsqueeze(2).repeat(1, 1, Nc)  # (B,1024,Nc)

        # combine
        feat = torch.cat([
            feat_global_expand,           # (B,1024,Nc)
            coarse_points.transpose(2,1), # (B,3,Nc)
            grid                          # (B,2,Nc)
        ], dim=1)

        fine = self.folding1(feat)
        return fine.transpose(2,1)        # → (B,Nc,3)
    def build_grid(self, B, P):
        s = int(math.sqrt(P)) + 1
        x = torch.linspace(-1, 1, steps=s)
        y = torch.linspace(-1, 1, steps=s)
        grid = torch.stack(torch.meshgrid(x, y, indexing='ij'), dim=0)
        grid = grid.reshape(2, -1)[:, :P]   # CẮT VỀ ĐÚNG P
        return grid.unsqueeze(0).repeat(B, 1, 1)

# ==================== HÀM HỖ TRỢ ====================
def normalize_pair(partial, complete):
    centroid = complete.mean(axis=0, keepdims=True)
    complete = complete - centroid
    partial = partial - centroid

    scale = np.max(np.linalg.norm(complete, axis=1))
    complete = complete / (scale + 1e-9)
    partial = partial / (scale + 1e-9)
    return partial, complete


def resample_points(pts, n):
    if len(pts) >= n:
        idx = np.random.choice(len(pts), n, replace=False)
    else:
        idx = np.random.choice(len(pts), n, replace=True)
    return pts[idx]


def create_hole(points):
    idx = np.random.randint(0, len(points))
    center = points[idx]
    dists = np.linalg.norm(points - center, axis=1)
    radius = np.percentile(dists, np.random.uniform(20, 50))
    return points[dists > radius]


def load_ply_file(file_path):
    mesh = trimesh.load(file_path, force='mesh')
    mesh.apply_translation(-mesh.center_mass)
    mesh.apply_scale(1.0 / np.max(mesh.extents))
    
    # Sample complete point cloud
    complete_points = mesh.sample(8192)
    
    # Create hole to get partial
    partial_points = create_hole(complete_points)
    
    # Normalize pair together (important!)
    partial_norm, complete_norm = normalize_pair(partial_points, complete_points)
    
    # Resample to fixed number
    partial_final = resample_points(partial_norm, 2048)
    gt_final = resample_points(complete_norm, 8192)
    
    return partial_final, gt_final

def infer(partial_np, model):
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(partial_np).float().unsqueeze(0).to(device)
        _, _, _, fine = model(x)
        return fine.cpu().numpy().squeeze()


def show_3d(partial, pred, gt):
    """Hiển thị 3 point clouds: input, predicted, ground truth"""
    fig = plt.figure(figsize=(18, 6))

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(partial[:,0], partial[:,1], partial[:,2], c='red', s=8, alpha=0.8)
    ax1.set_title("Input - Có lỗ", fontsize=14, fontweight='bold')
    ax1.set_axis_off()

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(pred[:,0], pred[:,1], pred[:,2], c='lime', s=4)
    ax2.set_title("Kết quả hoàn thiện (16k điểm)", fontsize=14, fontweight='bold')
    ax2.set_axis_off()

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(gt[:,0], gt[:,1], gt[:,2], c='dodgerblue', s=6, alpha=0.6)
    ax3.set_title("Ground Truth (mẫu gốc)", fontsize=14, fontweight='bold')
    ax3.set_axis_off()

    plt.suptitle("3D Point Cloud Hole Filling - PCN Model", fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ==================== GUI ====================
class App:
    def __init__(self):
        self.root = Tk()
        self.root.title("3D Hole Filling Demo - PCN Model")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        # Load model
        self.model = PCN().to(device)
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("Lỗi", f"Không tìm thấy model tại:\n{MODEL_PATH}")
            return
        
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
                print(f"  Training loss: {checkpoint.get('loss', 'unknown'):.6f}")
            else:
                self.model.load_state_dict(checkpoint)
                print("✓ Model loaded thành công!")
                
        except Exception as e:
            messagebox.showerror("Lỗi load model", f"Chi tiết:\n{str(e)}")
            return

        Label(self.root, text="Chọn file .ply từ ShapeNet để hoàn thiện lỗ:", 
              font=("Arial", 14, "bold"), bg="#f0f0f0").pack(pady=15)

        frame = tk.Frame(self.root)
        frame.pack(pady=10, padx=20, fill='both', expand=True)

        self.listbox = Listbox(frame, font=("Consolas", 11), height=18)
        scrollbar = Scrollbar(frame, orient="vertical")
        self.listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.listbox.yview)
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.load_files()

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=20)

        Button(btn_frame, text="HOÀN THIỆN ĐỐI TƯỢNG ĐÃ CHỌN", font=("Arial", 14, "bold"),
               bg="#4CAF50", fg="white", width=30, height=2,
               command=self.run_selected).pack(pady=10)

        Button(btn_frame, text="Chọn file .ply từ máy tính", font=("Arial", 11),
               bg="#2196F3", fg="white", width=30,
               command=self.run_custom).pack(pady=5)

        self.root.mainloop()

    def load_files(self):
        if not os.path.exists(DATA_FOLDER):
            messagebox.showerror("Lỗi", f"Không tìm thấy thư mục data:\n{DATA_FOLDER}")
            return
        
        ply_files = []
        for root, _, files in os.walk(DATA_FOLDER):
            for f in files:
                if f.lower().endswith('.ply'):
                    rel_path = os.path.relpath(os.path.join(root, f), DATA_FOLDER)
                    ply_files.append(rel_path)
        
        ply_files.sort()
        for f in ply_files[100:300]:
            self.listbox.insert(END, f)
        print(f"Đã load {len(ply_files)} file .ply")

    def run_selected(self):
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showwarning("Chưa chọn", "Vui lòng chọn một file từ danh sách!")
            return
        
        rel_path = self.listbox.get(sel[0])
        file_path = os.path.join(DATA_FOLDER, rel_path)
        self.process_file(file_path)

    def run_custom(self):
        file_path = filedialog.askopenfilename(filetypes=[("PLY files", "*.ply")])
        if file_path: 
            self.process_file(file_path)

    def process_file(self, ply_path):
        """Xử lý file .ply và hiển thị kết quả"""
        try:
            print(f"\n🔄 Đang xử lý: {ply_path}")
            
            partial, gt = load_ply_file(ply_path)
            print(f"  ✓ Loaded: partial={partial.shape}, gt={gt.shape}")
            
            predicted = infer(partial, self.model)
            print(f"  ✓ Inference: predicted={predicted.shape}")
            show_3d(partial, predicted, gt)
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Xử lý thất bại:\n{str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    App()