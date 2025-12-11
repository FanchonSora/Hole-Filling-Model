📌 README — Improved PCN for Point Cloud Completion
🚀 1. Giới thiệu

Project này xây dựng một mô hình Point Cloud Completion nhằm phục hồi mô hình 3D từ dữ liệu point cloud bị thiếu hụt. Mô hình được thiết kế theo hướng PCN (Point Completion Network) nhưng đã được cải tiến mạnh với kiến trúc hiện đại hơn, khả năng tái tạo chi tiết cao và độ sai số thấp.

Mục tiêu chính:

Hoàn thiện point cloud từ input sparse hoặc partial

Giảm Chamfer Distance, nâng F-Score

Tái tạo hình dạng mượt, ổn định, nhất quán

Đảm bảo inference hiệu quả, output dense

🧠 2. Kiến trúc Model

Model gồm ba phần chính:

(A) Encoder (PointNet + Transformer Fusion)

Sử dụng MLP + max-pooling để trích xuất weak local features

Kết hợp self-attention Transformers để mô hình hóa quan hệ không gian

Encoder output:

Vector global feature

Bộ local feature map

Output shape phù hợp cho decoding nhiều tầng

(B) Coarse Generator

Tạo coarse point cloud ban đầu (2,048 điểm) từ global feature

Sử dụng MLP nhiều tầng để học shape structure

Có vai trò định hình khối tổng thể

(C) Multi-Stage Refinement (Folding-based Upsampling)

Model sử dụng ba tầng refinement liên tiếp:

Fine1 (Patch Folding Stage 1)

Fine2 (Folding + Alignment)

Fine3 (Folding đa chiều + Residual Correction)

Các tầng folding:

Tạo lưới 2D (grid) quanh từng coarse point

Map lưới → không gian 3D thông qua feature toàn cục

Kết hợp residual learning để tăng độ chính xác

Output final: 16,384 điểm (hoặc theo cấu hình)

🧪 3. Loss Function

Model sử dụng nhiều loại loss để tối ưu đồng thời hình dạng, mật độ và độ mượt:

Chamfer Distance L1 (giữa coarse, fine1, fine2, fine3)

Repulsion Loss
Giảm clustering của điểm, cải thiện phân bố surface

Density Loss
Kiểm soát khoảng cách điểm → output mịn và dense hơn

Boundary Loss
Giúp tái tạo cạnh, đường cong, vùng mỏng

Tổng loss:

Loss = L_cd_total + λ1 * L_repulsion + λ2 * L_density + λ3 * L_boundary

📊 4. Kết quả Training (Summary)

Thông số đo được trên validation:

Metric	Value
Chamfer Distance	0.0295
EMD	0.0857
Hausdorff	0.09
Mean per-point error	0.0155
F-Score theo threshold:
Threshold	F-score
0.01	0.326
0.03	0.949
0.05	0.994
0.10	1.000

Lưu ý:
Point cloud trong dataset được scale theo bounding box ~1.8–1.9, nên threshold = 0.01 quá nhỏ.
F-score thực chất rất cao ở threshold hợp lý (0.03–0.05).

📈 5. Phân tích phân phối lỗi

Biểu đồ GT→Pred và Pred→GT cho thấy:

Đỉnh tập trung ở 0.01–0.015

Std nhỏ

95th percentile dưới 0.03

Không xuất hiện mode bất thường

👉 Điều này cho thấy model tái tạo surface rất ổn định, không bị lệch cấu trúc hay mất vùng.

🗂 6. Dataset

Dữ liệu được scale về bounding box có kích thước trung bình:
[1.79, 0.59, 1.87]

Input: partial point cloud (sparse/occluded)

Output: full point cloud (dense)

Chế độ sampling:

Coarse: 2048 điểm

Fine: 16384 điểm (gấp 8× qua patch folding)

⚙️ 7. Pipeline huấn luyện

Load dataset

Normalize & center object

Encoder tạo feature global

Generator sinh coarse point cloud

Ba tầng folding refinement → output dense

Tính toàn bộ loss

Tối ưu bằng AdamW

Cosine annealing scheduler + warmup

Log và evaluate theo mỗi epoch

🏎 8. Hiệu năng & Tối ưu

Training nhanh trên GPU RTX

Transformer + PointNet fusion nhẹ nhưng hiệu quả

Folding multi-stage → chất lượng cao nhưng inference vẫn nhanh

Có thể chạy real-time trong ứng dụng AR/VR hoặc robotics
🔮 9. Hướng phát triển tiếp theo

Thay coarse với Graph Convolutional Network

Thêm local patch attention

Áp dụng discriminator (GAN-based completion)

Huấn luyện multi-category hoặc shape-unified model

Export sang TensorRT phục vụ real-time robotics

📝 10. License

License Apache 2.0

🤝 11. Credits

Project phát triển bởi [Tên của bạn], dựa trên ý tưởng từ PCN, FoldingNet và các kiến trúc completion hiện đại.
