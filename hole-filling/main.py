import os
import math
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import trimesh
import open3d as o3d  # For visualization
import argparse
import kagglehub

# Config
CONFIG = {
    "num_complete_points": 2048,
    "num_partial_points": 2048,
    "feat_dim": 1024,
    "num_coarse": 2048,
    "num_fine": 16384,
    "fold_grid_size": 2,
    "train_epochs": 100,
    "batch_size": 6,
    "learning_rate": 1e-4,
    "warmup_epochs": 5,
    "scheduler": "cosine",
    "weight_chamfer": 2.0,
    "weight_repulsion": 1.0,
    "weight_density": 0.1,
    "weight_boundary": 0.05,
    "max_meshes": 10000,
    "hole_ratio_min": 0.2,
    "hole_ratio_max": 0.5,
}

# Dataset paths
local_path = "datasets/shapenetcorev2"  # Change to your dataset path
path = None
if os.path.exists(local_path):
    print("Found local ShapeNetCoreV2 directory!")
    path = local_path
else:
    print("Local directory not found. Downloading dataset...")
    path = kagglehub.dataset_download("hajareddagni/shapenetcorev2")
    print(path)
mesh_root = path

# Find PLY files
obj_paths = glob.glob(os.path.join(mesh_root, "**", "models", "*.ply"), recursive=True)
obj_paths = sorted(obj_paths)[:CONFIG["max_meshes"]]
print("Using PLY:", len(obj_paths))

def create_hole_in_pointcloud(complete_points, hole_ratio_min=0.2, hole_ratio_max=0.5):
    center_idx = np.random.randint(len(complete_points))
    center = complete_points[center_idx]
    hole_ratio = np.random.uniform(hole_ratio_min, hole_ratio_max)
    dists = np.linalg.norm(complete_points - center, axis=1)
    radius = np.percentile(dists, hole_ratio * 100)
    keep_mask = dists > radius
    partial_points = complete_points[keep_mask]
    if len(partial_points) < 100:
        keep_indices = np.argsort(dists)[100:]
        partial_points = complete_points[keep_indices]
    return partial_points

def sample_point_cloud_pair(obj_path, out_path):
    try:
        mesh = trimesh.load(obj_path, force='mesh')
        if hasattr(mesh, "repair"):
            mesh.repair.fix_normals()
            mesh.repair.fix_winding()
            mesh.repair.fill_holes()
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh.util.concatenate(mesh.geometry.values())
        mesh.apply_translation(-mesh.center_mass)
        mesh.apply_scale(1.0 / np.max(mesh.extents))
        complete_points = mesh.sample(CONFIG["num_complete_points"])
        partial_points = create_hole_in_pointcloud(
            complete_points, CONFIG["hole_ratio_min"], CONFIG["hole_ratio_max"]
        )
        np.savez_compressed(out_path, complete=complete_points.astype(np.float32), partial=partial_points.astype(np.float32))
    except Exception as e:
        print("Error at:", obj_path, "->", e)
        raise e

# Generate point clouds if not exist
os.makedirs("pointcloud_data", exist_ok=True)
pc_paths = []
for obj in tqdm(obj_paths, desc="Sampling Point Clouds"):
    name = os.path.splitext(os.path.basename(obj))[0]
    outp = f"pointcloud_data/{name}.npz"
    if not os.path.exists(outp):
        try:
            sample_point_cloud_pair(obj, outp)
        except Exception as e:
            print("Failed:", obj, e)
            continue
    pc_paths.append(outp)
print("Point cloud files:", len(pc_paths))

def normalize_pair(partial, complete):
    centroid = complete.mean(axis=0, keepdims=True)
    complete = complete - centroid
    partial = partial - centroid
    scale = np.max(np.linalg.norm(complete, axis=1))
    complete = complete / (scale + 1e-9)
    partial = partial / (scale + 1e-9)
    return partial, complete

class PointCloudCompletionDataset(Dataset):
    def __init__(self, npz_paths, num_partial=2048, num_complete=2048, augment=True):
        self.paths = npz_paths
        self.num_partial = num_partial
        self.num_complete = num_complete
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def _resample(self, pts, n, replace=True):
        N = pts.shape[0]
        if N == 0:
            return np.zeros((n, 3), dtype=np.float32)
        if N >= n:
            sel = np.random.choice(N, n, replace=False)
        else:
            sel = np.random.choice(N, n, replace=True)
        return pts[sel]

    def __getitem__(self, idx):
        data = np.load(self.paths[idx])
        partial = data['partial']
        complete = data['complete']
        partial, complete = normalize_pair(partial, complete)
        if self.augment:
            theta = np.random.uniform(0, 2 * np.pi)
            R = np.array([[np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta), np.cos(theta), 0],
                          [0, 0, 1]], dtype=np.float32)
            partial = partial.dot(R.T)
            complete = complete.dot(R.T)
        partial = self._resample(partial, self.num_partial)
        complete = self._resample(complete, self.num_complete)
        return torch.from_numpy(partial).float(), torch.from_numpy(complete).float()

# Model Components
class TransformerPointNetEncoder(nn.Module):
    def __init__(self, feat_dim=1024):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.self_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.1, batch_first=True)
        self.conv5 = nn.Conv1d(512, feat_dim, 1)
        self.bn5 = nn.BatchNorm1d(feat_dim)

    def forward(self, x):
        B, N, _ = x.shape
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x_attn = x.transpose(2, 1)
        x_attn, _ = self.self_attn(x_attn, x_attn, x_attn)
        x = x + x_attn.transpose(2, 1)
        feat_local = x.transpose(2, 1)
        x = self.bn5(self.conv5(x))
        feat_global = torch.max(x, dim=2)[0]
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
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.relu(self.gn2(self.conv2(x)))
        x = self.conv3(x)
        return x

class PCN(nn.Module):
    def __init__(self, feat_dim=1024, num_coarse=2048):
        super().__init__()
        self.encoder = TransformerPointNetEncoder(feat_dim)
        self.coarse_decoder = CoarseDecoder(feat_dim, num_coarse)
        self.folding1 = FoldingRefinement(feat_dim + 3 + 2)
        self.folding2 = FoldingRefinement(feat_dim + 3 + 2)
        self.folding3 = FoldingRefinement(feat_dim + 3 + 2)
        self.cross_attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=8, dropout=0.1, batch_first=True)
        self.refine_mlp = nn.Sequential(nn.Linear(feat_dim, 512), nn.ReLU(), nn.Linear(512, 3))
        self.coarse_feat_mlp = nn.Linear(3, feat_dim)
        self.project_partial_feat = nn.Linear(512, feat_dim)

    def forward(self, partial):
        feat_global, feat_local = self.encoder(partial)
        feat_local = self.project_partial_feat(feat_local)
        coarse = self.coarse_decoder(feat_global)
        coarse_ref = self.refine_with_attention(coarse, partial, feat_local)
        fine1 = self.expand_and_fold(feat_global, coarse_ref)
        fine2 = self.expand_and_fold(feat_global, fine1)
        fine3 = self.expand_and_fold(feat_global, fine2)
        return coarse, fine1, fine2, fine3

    def refine_with_attention(self, coarse, partial, partial_feat):
        B, Nc, _ = coarse.shape
        coarse_feat = self.coarse_feat_mlp(coarse)
        refined_feat, _ = self.cross_attn(query=coarse_feat, key=partial_feat, value=partial_feat)
        delta = self.refine_mlp(refined_feat)
        return coarse + delta

    def expand_and_fold(self, feat_global, coarse_points):
        B, Nc, _ = coarse_points.shape
        grid = self.build_grid(B, Nc).to(feat_global.device)
        feat_global_expand = feat_global.unsqueeze(2).repeat(1, 1, Nc)
        feat = torch.cat([feat_global_expand, coarse_points.transpose(2, 1), grid], dim=1)
        fine = self.folding1(feat)
        return fine.transpose(2, 1)

    def build_grid(self, B, P):
        s = int(math.sqrt(P)) + 1
        x = torch.linspace(-1, 1, steps=s)
        y = torch.linspace(-1, 1, steps=s)
        grid = torch.stack(torch.meshgrid(x, y, indexing='ij'), dim=0)
        grid = grid.reshape(2, -1)[:, :P]
        return grid.unsqueeze(0).repeat(B, 1, 1)

# Loss Functions
def chamfer_distance(pred, gt):
    dmat = torch.cdist(pred, gt, p=2)
    d1 = torch.min(dmat, dim=2)[0]
    d2 = torch.min(dmat, dim=1)[0]
    return (d1.mean(dim=1) + d2.mean(dim=1)).mean()

def repulsion_loss(pred, partial=None, k=10, h=0.07):
    B, N, _ = pred.shape
    rep_all = 0.0
    for b in range(B):
        idx = torch.cdist(pred[b], pred[b]).topk(k+1, largest=False, dim=1)[1][:, 1:]
        neigh = pred[b][idx]
        center = pred[b].unsqueeze(1)
        d = torch.norm(center - neigh, dim=2)
        rep = F.relu(h - d)
        if partial is not None:
            dist_in = torch.cdist(pred[b], partial[b])
            weight = 1 + 3 * (dist_in.min(dim=1)[0] > 0.08).float()
            rep = rep * weight.unsqueeze(-1)
        rep_all += rep.mean()
    return rep_all / B

def density_loss(pred, gt, bandwidth=0.02, sample_ratio=0.1):
    B, M, _ = gt.shape
    S = max(32, int(M * sample_ratio))
    idx = torch.randint(0, M, (B, S), device=pred.device)
    sample = torch.gather(gt, 1, idx.unsqueeze(-1).repeat(1, 1, 3))
    d_pred = torch.cdist(sample, pred)
    d_gt = torch.cdist(sample, gt)
    k_pred = torch.exp(-d_pred**2 / (2 * bandwidth**2)).mean(dim=2)
    k_gt = torch.exp(-d_gt**2 / (2 * bandwidth**2)).mean(dim=2)
    return ((k_pred - k_gt)**2).mean()

def boundary_aware_loss(pred_fine, partial, alpha=2.0):
    dist_to_input = torch.cdist(pred_fine, partial)
    min_dist = torch.min(dist_to_input, dim=2)[0]
    boundary_penalty = F.relu(min_dist - 0.1).pow(alpha)
    return boundary_penalty.mean()

def combined_loss(pred_coarse, pred_fine1, pred_fine2, pred_fine3, gt, partial, weights=None):
    if weights is None:
        weights = CONFIG
    loss_chamfer = (
        0.1 * chamfer_distance(pred_coarse, gt) +
        0.2 * chamfer_distance(pred_fine1, gt) +
        0.3 * chamfer_distance(pred_fine2, gt) +
        1.0 * chamfer_distance(pred_fine3, gt)
    )
    loss_rep = repulsion_loss(pred_fine3, partial=partial, k=10, h=0.05)
    loss_den = density_loss(pred_fine3, gt, bandwidth=0.015, sample_ratio=0.3)
    loss_boundary = boundary_aware_loss(pred_fine3, partial, alpha=2.0)
    total = (
        weights['weight_chamfer'] * loss_chamfer +
        weights['weight_repulsion'] * loss_rep +
        weights['weight_density'] * loss_den +
        weights['weight_boundary'] * loss_boundary
    )
    return total, {'chamfer': loss_chamfer.item(), 'repulsion': loss_rep.item(), 'density': loss_den.item(), 'boundary': loss_boundary.item()}

# Training Function
def train_model(model, dataloader, epochs=100, device='cuda'):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-4)
    warmup_lambda = lambda epoch: (epoch + 1) / CONFIG["warmup_epochs"] if epoch < CONFIG["warmup_epochs"] else 1.0
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - CONFIG["warmup_epochs"], eta_min=1e-6)
    scaler = GradScaler(enabled=True)
    best_loss = float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        steps = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for partial, complete in loop:
            partial, complete = partial.to(device), complete.to(device)
            optimizer.zero_grad()
            with autocast(enabled=True):
                coarse, fine1, fine2, fine3 = model(partial)
                loss, _ = combined_loss(coarse, fine1, fine2, fine3, complete, partial)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            steps += 1
            loop.set_postfix({"Loss": f"{epoch_loss/steps:.4f}"})
        if epoch <= CONFIG["warmup_epochs"]:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()
        avg_loss = epoch_loss / steps
        print(f"Epoch {epoch}: Loss = {avg_loss:.6f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'pcn_best.pth')
            print(f"✓ Saved best model (loss: {best_loss:.6f})")

# Inference Function
def infer(model, partial_path, device='cuda', visualize=True):
    model.eval()
    data = np.load(partial_path)
    partial = data['partial']
    complete = data['complete']  # For comparison
    partial, _ = normalize_pair(partial, complete)  # Normalize using complete for fairness
    partial = np.random.choice(partial.shape[0], CONFIG["num_partial_points"], replace=False) if partial.shape[0] > CONFIG["num_partial_points"] else partial
    partial = torch.from_numpy(partial).float().unsqueeze(0).to(device)
    with torch.no_grad():
        _, _, _, fine3 = model(partial)
    fine3 = fine3.cpu().numpy().squeeze()
    print("Inference done. Output shape:", fine3.shape)
    if visualize:
        visualize_point_cloud(partial.cpu().numpy().squeeze(), fine3, complete)
    return fine3

# Visualize Function (for CG demo)
def visualize_point_cloud(partial, predicted, ground_truth=None):
    pcd_partial = o3d.geometry.PointCloud()
    pcd_partial.points = o3d.utility.Vector3dVector(partial)
    pcd_partial.paint_uniform_color([1, 0, 0])  # Red for partial

    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(predicted)
    pcd_pred.paint_uniform_color([0, 1, 0])  # Green for predicted

    geometries = [pcd_partial, pcd_pred]
    if ground_truth is not None:
        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(ground_truth)
        pcd_gt.paint_uniform_color([0, 0, 1])  # Blue for GT
        geometries.append(pcd_gt)
    
    o3d.visualization.draw_geometries(geometries)

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCN Project")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'infer'], help='Mode: train or infer')
    parser.add_argument('--infer_path', type=str, default='pointcloud_data/model_0000.npz', help='Path to .npz for inference')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PCN(feat_dim=CONFIG["feat_dim"], num_coarse=CONFIG["num_coarse"])

    if args.mode == 'train':
        ds = PointCloudCompletionDataset(pc_paths, augment=True)
        dl = DataLoader(ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2)
        train_model(model, dl, epochs=CONFIG["train_epochs"], device=device)
    elif args.mode == 'infer':
        model.load_state_dict(torch.load('pcn_best.pth', map_location=device))
        infer(model, args.infer_path, device=device)