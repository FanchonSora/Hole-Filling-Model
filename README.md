# Hole-Filling Model — Improved PCN for Point Cloud Completion
/ Mô hình hoàn thiện point cloud — Phiên bản PCN cải tiến

---

## English — Overview

This project implements an improved Point Completion Network (PCN) to reconstruct dense 3D point clouds from partial or sparse inputs. The model produces a coarse-to-fine output, using a PointNet + Transformer encoder, a coarse generator, and multi-stage folding-based refinement to yield high-quality dense reconstructions suitable for AR/VR and robotics.

Goals:
- Complete partial point clouds into dense, consistent surfaces
- Reduce Chamfer Distance and improve F-Score
- Produce smooth, stable, and inference-efficient outputs

---

## Tiếng Việt — Tổng quan

Project này xây dựng một mô hình hoàn thiện point cloud (PCN cải tiến) nhằm khôi phục mô hình 3D đầy đủ từ dữ liệu bị thiếu. Mô hình hoạt động theo cơ chế coarse → multi-stage refinement để tạo output dense, phù hợp cho ứng dụng AR/VR và robotics.

Mục tiêu:
- Hoàn thiện point cloud đầu vào (sparse/partial) thành output dense
- Giảm Chamfer Distance, tăng F-Score
- Kết quả mượt, ổn định, inference nhanh

---

## Architecture / Kiến trúc

1. Encoder — PointNet + Transformer Fusion
   - MLP + max-pooling để trích xuất local features
   - Self-attention Transformer layers để mô hình hóa mối quan hệ không gian
   - Outputs: global feature vector + local feature map

2. Coarse Generator
   - MLP-based module sinh coarse point cloud (default 2048 points)
   - Hình thành cấu trúc khối tổng thể của shape

3. Multi-Stage Refinement (Folding-based Upsampling)
   - Ba tầng refinement: Fine1 → Fine2 → Fine3
   - Mỗi tầng tạo lưới 2D quanh coarse point, map sang không gian 3D với features
   - Kết hợp residual correction để tăng chính xác
   - Final output configurable (default 16384 points)

---

## Losses / Hàm mất mát

- Chamfer Distance (L1) — áp dụng ở tất cả các cấp (coarse, fine1, fine2, fine3)
- Repulsion Loss — giảm clustering của điểm
- Density Loss — điều chỉnh mật độ, đảm bảo output mịn
- Boundary Loss — khuyến khích tái tạo các cạnh và vùng mỏng

Total loss:
Loss = L_cd_total + λ1 * L_repulsion + λ2 * L_density + λ3 * L_boundary

---

## Dataset & Preprocessing / Dữ liệu và tiền xử lý

- Input: partial point cloud (sparse/occluded)
- Output: full point cloud (dense)
- Normalization: center và scale theo bounding box
- Typical bounding box scale ≈ [1.79, 0.59, 1.87]
- Sampling:
  - Coarse: 2048 points
  - Fine: 16384 points (8× via patch folding)

---

## Training pipeline / Quy trình huấn luyện

- Load and preprocess dataset
- Encoder extracts global + local features
- Coarse generator produces initial shape
- Apply three folding refinement stages → final dense output
- Compute combined losses and optimize with AdamW
- Scheduler: cosine annealing + warmup
- Logging and evaluation per epoch

---

## Results (Validation Summary) / Kết quả (Tổng hợp trên validation)

Metrics:
- Chamfer Distance: 0.0295
- EMD: 0.0857
- Hausdorff: 0.09
- Mean per-point error: 0.0155

F-Score:
- threshold 0.01 → 0.326
- threshold 0.03 → 0.949
- threshold 0.05 → 0.994
- threshold 0.10 → 1.000

Note: dataset scaled to bounding boxes ~1.8–1.9; very small thresholds (e.g. 0.01) may be strict.

Phân tích lỗi:
- Distribution peaks around 0.01–0.015
- Small std, 95th percentile < 0.03
- No abnormal modes → reconstructed surfaces are stable and consistent

---

## Performance & Optimization / Hiệu năng & tối ưu

- Fast training and inference on modern GPUs (e.g., NVIDIA RTX)
- Lightweight Transformer + PointNet fusion for efficiency
- Multi-stage folding yields high-quality results while keeping inference efficient
- Candidate for near real-time use in AR/VR and robotics

---

## How to use / Hướng dẫn nhanh

1. Requirements:
   - Python 3.8+
   - PyTorch (compatible version)
   - Dependencies: numpy, tqdm, open3d (optional for visualization), etc.

2. Basic steps:
   - Prepare dataset and config (paths, sampling sizes, λ weights)
   - Train: python train.py --config configs/train.yaml
   - Evaluate: python eval.py --checkpoint path/to/checkpoint
   - Visualize outputs with Open3D or your preferred viewer

3. Configurable options:
   - Number of output points
   - Learning rates, scheduler, loss weights (λ1, λ2, λ3)
   - Batch size, augmentation settings

---

## Future work / Hướng phát triển

- Replace coarse generator with a Graph Convolutional Network
- Add local patch attention mechanisms
- Explore adversarial training (GAN-based completion)
- Train a multi-category / shape-unified model
- Export model to TensorRT for real-time robotics deployment

---

## License / Giấy phép

Apache License 2.0

---

## Credits / Tác giả & Tham khảo

Developed by [Your Name] (replace with your name). Inspired by PCN, FoldingNet and modern completion architectures.

If you want, I can:
- Commit this README directly to the repo (I'll need repo push permissions or your confirmation to run the update), or
- Further customize wording, add usage examples, config samples, or badges.
