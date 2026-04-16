# 📁 Phân Tích Chi Tiết Các File Trong Thư Mục `denoising_diffusion_pytorch`

## 🎯 Tổng Quan

Thư mục này chứa **18 file Python** triển khai các biến thể khác nhau của Diffusion Models. Các file được chia thành 4 nhóm chính:

1. **Core Implementation** - Các triển khai diffusion cơ bản
2. **Variants** - Các biến thể và cải tiến
3. **Architecture** - Các mô hình mạng nơ-ron
4. **Utilities** - Công cụ hỗ trợ

---

## 📊 Chi Tiết Từng File

### **NHÓM 1: CORE IMPLEMENTATION (Triển khai cơ bản)**

#### **1. `__init__.py`**
- **Tác dụng:** File khởi tạo package, export các API chính
- **Nội dung:**
  - Import tất cả các class và model từ các file khác
  - Cung cấp điểm truy cập dễ dàng cho người dùng
  - Export: `GaussianDiffusion`, `Unet`, `Trainer` (2D)
  - Export: `GaussianDiffusion1D`, `Unet1D`, `Trainer1D` (1D)
  - Export: Các biến thể khác như `LearnedGaussianDiffusion`, `ContinuousTimeGaussianDiffusion`, `KarrasUnet`, v.v.
- **Ví dụ sử dụng:**
  ```python
  from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
  ```

---

#### **2. `denoising_diffusion_pytorch.py`** ⭐ **[FILE CHÍNH]**
- **Tác dụng:** Triển khai đầy đủ DDPM (Denoising Diffusion Probabilistic Models) cho ảnh 2D
- **Kích thước:** 1128 dòng code
- **Các thành phần chính:**

  **A. Helper Functions & Utilities:**
  - `exists()`, `default()`, `cast_tuple()` - Hàm tiện ích chung
  - `normalize_to_neg_one_to_one()` - Chuẩn hóa ảnh từ [0,1] → [-1,1]
  - `extract()` - Trích xuất giá trị từ schedule tại timestep t

  **B. Architecture Components:**
  - `Block` - Khối cơ bản (Conv2d + RMSNorm + SiLU)
  - `ResnetBlock` - Khối Residual với Time Embedding
  - `LinearAttention` - Attention hiệu quả (O(n²) → O(n))
  - `Attention` - Full Self-Attention (có thể dùng Flash Attention)
  - `SinusoidalPosEmb` - Sinusoidal positional embedding
  - `RandomOrLearnedSinusoidalPosEmb` - Random/Learned sinusoidal

  **C. Core Model:**
  - `Unet` - U-Net architecture:
    - Encoder (downsampling) → Bottleneck → Decoder (upsampling)
    - Input: (B, C, H, W) + timestep
    - Output: (B, C, H, W) dự đoán noise/x₀/v
    - Có support cho self-conditioning

  **D. Diffusion Process:**
  - `GaussianDiffusion` - Quy trình Gaussian Diffusion:
    - Beta schedule: linear, cosine, sigmoid
    - Forward process: q(x_t|x₀) = thêm noise từng bước
    - Reverse process: p(x_{t-1}|x_t) = bỏ noise từng bước
    - Mục tiêu dự đoán: pred_noise, pred_x0, pred_v

  **E. Dataset & Training:**
  - `Dataset` - Load ảnh từ folder, áp dụng transform
  - `Trainer` - Lớp huấn luyện:
    - Hỗ trợ multi-GPU qua Accelerator
    - Mixed precision (FP16 autocast)
    - EMA (Exponential Moving Average) cho model
    - Tính FID Score tự động
    - Checkpoint & sampling định kỳ

- **Quy trình Training:**
  ```
  1. Lấy ảnh gốc x₀
  2. Sample timestep t ngẫu nhiên
  3. Thêm noise: x_t = √ᾱ_t·x₀ + √(1-ᾱ_t)·ε
  4. Model dự đoán: ŷ = UNet(x_t, t)
  5. Tính Loss: MSE(ŷ, target)
  6. Backward & optimize
  ```

---

#### **3. `version.py`**
- **Tác dụng:** Quản lý phiên bản package
- **Nội dung:** `__version__ = '2.2.5'`
- **Mục đích:** Cho phép tracking version khi deploy

---

### **NHÓM 2: BIẾN THỂ & CẢI TIẾN (Diffusion Variants)**

#### **4. `classifier_free_guidance.py`**
- **Tác dụng:** Triển khai Classifier-Free Guidance (CFG) cho conditional generation
- **Kích thước:** 854 dòng
- **Khác biệt chính:**
  - Cho phép điều khiển quá trình generation bằng text/class label
  - Kết hợp conditional và unconditional predictions
  - Công thức: `ŷ = ŷ_uncond + scale * (ŷ_cond - ŷ_uncond)`
  - Ứng dụng: Text-to-Image, Class-conditional generation
- **Ví dụ:** DALL-E 2, Imagen sử dụng CFG

---

#### **5. `learned_gaussian_diffusion.py`**
- **Tác dụng:** Gaussian Diffusion với variance được học (Learned Variance)
- **Kích thước:** 156 dòng
- **Khác biệt:**
  - Thay vì variance cố định, model học dự đoán variance
  - Model output gồm 3 phần: noise, x₀, variance
  - Tính KL divergence giữa posterior và predicted
- **Lợi ích:** Cải thiện chất lượng tại timesteps đầu

---

#### **6. `weighted_objective_gaussian_diffusion.py`**
- **Tác dụng:** Weighted Objective - Model học cả pred_noise lẫn pred_x0 với weight học được
- **Kích thước:** 83 dòng
- **Cách hoạt động:**
  - Model output: [pred_noise, pred_x0, weights]
  - Tính toán cả 2 predictions
  - Lấy weighted sum: `x_start = w₁·x₀_from_noise + w₂·x₀_direct`
- **Lợi ích:** Tự động cân bằng giữa 2 objectives

---

#### **7. `continuous_time_gaussian_diffusion.py`**
- **Tác dụng:** Diffusion với time liên tục (Continuous Time), không rời rạc
- **Kích thước:** 276 dòng
- **Khác biệt:**
  - Timestep t ∈ [0, 1] (không phải [0, 1000])
  - Beta schedule là hàm liên tục
  - ODE/SDE based sampling
- **Tham khảo:** Song et al., "Score-based Generative Modeling through Stochastic Differential Equations"

---

#### **8. `v_param_continuous_time_gaussian_diffusion.py`**
- **Tác dụng:** V-parameterization cho Continuous Time Diffusion
- **Kích thước:** 187 dòng
- **Đặc điểm:**
  - Sử dụng v-space thay vì ε-space
  - Cosine schedule cho log-SNR
  - Ổn định hơn khi training
- **Tham khảo:** "Diffusion Models Beat GANs on Image Synthesis" (Dhariwal & Nichol)

---

#### **9. `elucidated_diffusion.py`**
- **Tác dụng:** Elucidated Diffusion từ paper "Elucidating the Design Space of Diffusion-Based Generative Models"
- **Kích thước:** 278 dòng
- **Khác biệt:**
  - Parametrization khác với σ (sigma noise level) thay vì timestep
  - Sigma range: [σ_min, σ_max]
  - Stochastic/Deterministic sampling tuỳ chọn
  - Ứng dụng trong EDM (Elucidated Diffusion Models)

---

### **NHÓM 3: KIẾN TRÚC MẠNG (Network Architectures)**

#### **10. `karras_unet.py`** ⭐
- **Tác dụng:** U-Net Magnitude-Preserving từ Karras et al. 2023
- **Kích thước:** 723 dòng
- **Đặc điểm chính:**
  - `MPSiLU` - Magnitude-Preserving SiLU activation
  - `Gain` - Layer scaling parameter
  - `MPCat` - Concatenation bảo toàn magnitude
  - Ổn định training hơn U-Net thường
  - Hỗ trợ InvSqrtDecayLRSched (Learning Rate Scheduler)
- **Ứng dụng:** Model hiện đại cho high-quality generation

---

#### **11. `karras_unet_1d.py`**
- **Tác dụng:** Phiên bản 1D của Karras U-Net
- **Kích thước:** 715 dòng
- **Áp dụng:** Dữ liệu 1D (Audio, Time Series, Sequences)
- **Khác biệt:** Conv1d thay vì Conv2d, Pooling 1D

---

#### **12. `karras_unet_3d.py`**
- **Tác dụng:** Phiên bản 3D của Karras U-Net
- **Kích thước:** 852 dòng
- **Áp dụng:** Dữ liệu 3D (Video, Medical Imaging, 3D Objects)
- **Khác biệt:** Conv3d, Attention 3D, hỗ trợ Optional parameters

---

#### **13. `simple_diffusion.py`**
- **Tác dụng:** Triển khai đơn giản, dễ hiểu cho sinh viên
- **Kích thước:** 706 dòng
- **Đặc điểm:**
  - Có U-ViT (Vision Transformer based U-Net)
  - SimpleDiffusion class - phiên bản minimal
  - Tốt cho learning/debugging
- **Use case:** Giáo dục, prototyping nhanh

---

### **NHÓM 4: 1D IMPLEMENTATION**

#### **14. `denoising_diffusion_pytorch_1d.py`**
- **Tác dụng:** Triển khai đầy đủ cho dữ liệu 1D (tương tự `denoising_diffusion_pytorch.py` nhưng cho 1D)
- **Kích thước:** 923 dòng
- **Thành phần:**
  - `Unet1D` - U-Net 1D với Conv1d
  - `GaussianDiffusion1D` - Diffusion process cho 1D
  - `Trainer1D` - Trainer cho 1D
  - `Dataset1D` - Load audio/time-series data
- **Ứng dụng:** Music generation, Audio synthesis, Time series forecasting

---

### **NHÓM 5: CONDITIONAL GENERATION & IMAGE EDITING**

#### **15. `guided_diffusion.py`**
- **Tác dụng:** Diffusion với Guidance (hướng dẫn generation)
- **Kích thước:** 1017 dòng
- **Loại Guidance:**
  - Classifier Guidance - dùng classifier để guide
  - Feature Guidance - guide bằng low-level features
- **Ứng dụng:** Điều khiển tính chất ảnh sinh (color, texture, style)

---

#### **16. `repaint.py`**
- **Tác dụng:** Inpainting & Image Editing qua Diffusion
- **Kích thước:** 1141 dòng
- **Chức năng:**
  - Điền vào vùng masked (inpainting)
  - Chỉnh sửa ảnh có điều kiện
  - Thay đổi semantic của ảnh
  - Công thức: `x_t ← blend(predicted_x_t, original_x_t, mask)`
- **Ứng dụng:** Photo editing, Object removal, Image restoration

---

### **NHÓM 6: UTILITIES & TOOLS**

#### **17. `attend.py`** 🔧
- **Tác dụng:** Wrapper thống nhất cho Attention mechanisms
- **Kích thước:** 148 dòng
- **Hỗ trợ:**
  - Flash Attention (PyTorch 2.0+) - cực nhanh
  - Efficient Attention (memory optimized)
  - Math Attention (fallback)
  - Tự động chọn backend tối ưu cho GPU
- **Mục đích:** Tối ưu hóa tốc độ inference và training

**Chi tiết:**
```python
class Attend(nn.Module):
    def __init__(self, flash=False):
        # flash=True: dùng Flash Attention (nhanh 2-3x)
        # Tự động detect GPU: A100 → FLASH, khác → EFFICIENT
```

---

#### **18. `fid_evaluation.py`** 📊
- **Tác dụng:** Tính Frechet Inception Distance (FID Score)
- **Kích thước:** 143 dòng
- **Công dụng:**
  - Đo chất lượng ảnh sinh (so với real images)
  - FID càng thấp càng tốt (< 5 là excellent)
  - Tự động cache dataset stats
  - Sử dụng Inception V3 pretrained
- **Công thức:**
  ```
  FID = ||m_real - m_fake||² + Tr(Σ_real + Σ_fake - 2(Σ_real·Σ_fake)^0.5)
  ```

**Workflow FID:**
```
1. Extract features từ real images → m_real, Σ_real
2. Extract features từ generated images → m_fake, Σ_fake
3. Tính Frechet distance
4. Lower FID = Better quality
```

---

## 📋 Bảng Tóm Tắt

| File | Loại | Dòng | Tác Dụng | Mục Đích |
|------|------|------|---------|---------|
| `__init__.py` | Core | ~50 | Export API | Package initialization |
| `version.py` | Core | ~5 | Version tracking | Package versioning |
| **`denoising_diffusion_pytorch.py`** | **Core** | **1128** | **DDPM 2D** | **Main implementation** |
| `classifier_free_guidance.py` | Variant | 854 | CFG guidance | Conditional generation |
| `learned_gaussian_diffusion.py` | Variant | 156 | Learned variance | Improved quality |
| `weighted_objective_gaussian_diffusion.py` | Variant | 83 | Weighted objectives | Balanced objectives |
| `continuous_time_gaussian_diffusion.py` | Variant | 276 | Continuous time | ODE/SDE sampling |
| `v_param_continuous_time_gaussian_diffusion.py` | Variant | 187 | V-parameterization | Stable training |
| `elucidated_diffusion.py` | Variant | 278 | Elucidated DDPM | Design space exploration |
| **`karras_unet.py`** | **Architecture** | **723** | **Magnitude-Preserving UNet** | **Modern architecture** |
| `karras_unet_1d.py` | Architecture | 715 | 1D MP-UNet | Audio/sequences |
| `karras_unet_3d.py` | Architecture | 852 | 3D MP-UNet | Video/3D objects |
| `simple_diffusion.py` | Architecture | 706 | Simplified DDPM | Learning/quick prototyping |
| `denoising_diffusion_pytorch_1d.py` | 1D | 923 | Full DDPM 1D | Audio/time-series |
| `guided_diffusion.py` | Conditional | 1017 | Guided generation | Controlled sampling |
| `repaint.py` | Conditional | 1141 | Inpainting/editing | Image manipulation |
| `attend.py` | Utility | 148 | Attention wrapper | Performance optimization |
| `fid_evaluation.py` | Utility | 143 | FID scoring | Quality evaluation |

---

## 🎓 Mối Quan Hệ Giữa Các File

```
                    __init__.py (Entry point)
                          │
                ┌─────────┼─────────┐
                │         │         │
        Version.py   Main Core   Variants
                │         │         │
                └────┬────┴────┬────┘
                     │         │
              Architectures  Utilities
                  │              │
        ┌─────────┼─────┐   ┌────┴─────┐
        │         │     │   │          │
    2D/1D/3D   Simple  MP  Attend  FID_Eval
    (3 files) Diffusion Unet (3 files)
                         
```

---

## 🚀 Workflow Điển Hình

### **Scenario 1: Sinh ảnh 2D cơ bản (Quickstart)**
```python
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(dim=64, dim_mults=(1,2,4,8))
diffusion = GaussianDiffusion(model, image_size=32, timesteps=1000)
trainer = Trainer(diffusion, folder="./images", train_num_steps=100000)
trainer.train()  # → denoising_diffusion_pytorch.py
```

### **Scenario 2: Text-to-Image với Guidance**
```python
from denoising_diffusion_pytorch import classifier_free_guidance

# → classifier_free_guidance.py
diffusion = classifier_free_guidance.ClassifierFreeGuidance(...)
images = diffusion.sample_with_guidance(text_prompts, guidance_scale=7.5)
```

### **Scenario 3: Inpainting ảnh**
```python
from denoising_diffusion_pytorch import repaint

editor = repaint.Repaint(...)  # → repaint.py
result = editor.inpaint(image, mask, num_steps=50)
```

### **Scenario 4: Audio synthesis (1D)**
```python
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D

model = Unet1D(dim=64)
diffusion = GaussianDiffusion1D(model, length=16000)
trainer = Trainer1D(diffusion, ...)  # → denoising_diffusion_pytorch_1d.py
```

### **Scenario 5: Tính FID Score để đánh giá**
```python
from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation

fid = FIDEvaluation(batch_size=32, dl=dataloader, ...)  # → fid_evaluation.py
score = fid.fid_score()  # Lower is better
```

---

## 💡 Khuyến Nghị Sử Dụng

| Nhu Cầu | File Chính | Ghi Chú |
|--------|-----------|--------|
| **Học/Demo** | `simple_diffusion.py` | Dễ hiểu, code clean |
| **Sinh ảnh 2D chất lượng** | `denoising_diffusion_pytorch.py` + `karras_unet.py` | SOTA |
| **Điều khiển generation** | `classifier_free_guidance.py` | Text-to-image |
| **Chỉnh sửa ảnh** | `repaint.py` | Inpainting, editing |
| **Audio/Music** | `denoising_diffusion_pytorch_1d.py` | Music generation |
| **Video/3D** | `karras_unet_3d.py` + `guided_diffusion.py` | Complex tasks |
| **Đánh giá chất lượng** | `fid_evaluation.py` | Quality metrics |
| **Tối ưu tốc độ** | `attend.py` | Flash Attention |

---

## 📚 Tham Khảo Paper

- **DDPM**: Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
- **Guidance**: Ho & Salimans, "Classifier-Free Diffusion Guidance" (2021)
- **Continuous Time**: Song et al., "Score-Based Generative Modeling through SDEs" (2021)
- **V-parameterization**: Salimans & Ho, "Progressive Distillation for Fast Sampling" (2022)
- **Elucidated**: Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models" (2022)
- **MP-UNet**: Karras et al., "Analyzing and Improving the Image Quality of StyleGAN" (2023)
- **Inpainting**: Song et al., "Diffusion Models as Plug-and-Play Priors" (2022)
