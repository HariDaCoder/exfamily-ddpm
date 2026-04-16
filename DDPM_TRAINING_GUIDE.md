# 🎓 Hướng Dẫn Chi Tiết: DDPM Training & Noise Customization

## 📖 Mục Lục
1. [Cách Model Học](#cách-model-học)
2. [Nơi Thay Đổi Noise Distribution](#nơi-thay-đổi-noise-distribution)
3. [Giải Thích Từng Biến](#giải-thích-từng-biến)
4. [Ví Dụ Thực Hành](#ví-dụ-thực-hành)

---

## 🧠 Cách Model Học

### Forward Process (Training)

```
Bước 1: Lấy batch từ dataset
   x₀ = [sample1, sample2, ..., sample64] ∈ [-1, 1]
   
Bước 2: Sample timestep ngẫu nhiên
   t ~ Uniform(0, 100)
   
Bước 3: Sample noise từ phân phối
   ε ~ N(0, I)  # Hoặc phân phối khác
   
Bước 4: Tính x_t bằng forward diffusion
   x_t = √ᾱ_t * x₀ + √(1-ᾱ_t) * ε
   
   Giải thích:
   - √ᾱ_t:           hệ số signal (càng cao t, càng nhỏ)
   - √(1-ᾱ_t):       hệ số noise  (càng cao t, càng lớn)
   - Ở t=0:  x_t ≈ x₀ (gần như data gốc)
   - Ở t=T:  x_t ≈ ε (gần như pure noise)
   
Bước 5: Model dự đoán noise
   ε_pred = UNet(x_t, t)
   
   Model nhận input:
   - x_t: dữ liệu nhiễu
   - t:   timestep (để biết mình ở stage nào)
   
   Output: dự đoán về noise được thêm
   
Bước 6: Tính loss (MSE)
   L = ||ε_pred - ε||²
   
   Training sẽ cố gắng làm ε_pred gần với ε thực
   
Bước 7: Backward propagation
   ∇θ L được tính
   θ ← θ - α * ∇θ L
```

### Reverse Process (Inference)

```
Bước 1: Bắt đầu từ pure noise
   x_T ~ N(0, I)
   
Bước 2: Lặp từ t=T-1 xuống t=0
   
   Lặp t=99 → 0:
      - Dùng trained model dự đoán noise: ε_pred = UNet(x_t, t)
      - Tính x_{t-1} từ x_t bằng công thức reverse
      - Thêm một chút noise (trừ khi t=0)
   
Bước 3: Kết quả: x_0
   x_0 là mẫu sinh ra mới
```

---

## 🔧 Nơi Thay Đổi Noise Distribution

### ⭐ Cách 1: Beta Schedule (Quan Trọng Nhất)

Beta schedule kiểm soát **mức độ nhiễu được thêm ở mỗi bước**.

#### Vị Trí Code:

```python
# File: Notebook cell "GaussianDiffusion1D Class"
# Line: diffusion = GaussianDiffusion1D(...)

diffusion = GaussianDiffusion1D(
    model,
    seq_length=seq_length,
    timesteps=100,
    beta_schedule='cosine',  # ← THAY ĐỔI ĐÂY
    objective='pred_noise'
)
```

#### Các Tùy Chọn:

| Schedule | Công Thức | Đặc Điểm | Khi Nào Dùng |
|----------|-----------|----------|------------|
| **linear** | β_t = β_start + (β_end - β_start) * t/T | Đơn giản, tuyến tính | Baseline, debug |
| **cosine** | β_t từ cosine curve | Giữ signal lâu hơn | Thường dùng, tốt |
| **sigmoid** | β_t từ sigmoid curve | Smooth transition | Modern, SOTA |

#### So Sánh:

```
t=0 (bắt đầu):
  - Linear:   β ≈ 0.0001      (thêm ít noise)
  - Cosine:   β ≈ 0.00005     (thêm ít noise, ít hơn)
  - Sigmoid:  β ≈ 0.00001     (thêm ít noise nhất)

t=50 (giữa):
  - Linear:   β ≈ 0.01        (thêm nhiều noise)
  - Cosine:   β ≈ 0.0085      (thêm nhiều noise, ít hơn)
  - Sigmoid:  β ≈ 0.005       (thêm noise vừa phải)

t=99 (cuối):
  - Linear:   β ≈ 0.02        (thêm rất nhiều noise)
  - Cosine:   β ≈ 0.019       (thêm rất nhiều noise)
  - Sigmoid:  β ≈ 0.019       (thêm rất nhiều noise)
```

---

### ⭐ Cách 2: Loại Noise Distribution

Mặc định là **Gaussian** (N(0, I)), nhưng có thể dùng phân phối khác.

#### Vị Trí Code:

```python
# File: Notebook cell "GaussianDiffusion1D Class"
# Function: q_sample(self, x_start, t, noise=None)

def q_sample(self, x_start, t, noise=None):
    """Forward diffusion"""
    if noise is None:
        noise = torch.randn_like(x_start)  # ← THAY ĐỔI ĐÂY
    
    sqrt_alpha_prod = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alpha_prod = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

    return sqrt_alpha_prod * x_start + sqrt_one_minus_alpha_prod * noise
```

#### Các Phân Phối:

```python
# Gaussian (Default) - Smooth, standard
noise = torch.randn_like(x_start)

# Uniform [-1, 1]
noise = (torch.rand_like(x_start) * 2 - 1) * math.sqrt(3)

# Exponential
noise = torch.exponential_(torch.ones_like(x_start)) - 1

# Laplace
noise = torch.randn_like(x_start) + torch.randn_like(x_start)

# Custom mixture
r = torch.rand_like(x_start)
noise = torch.where(r < 0.8, 
                    torch.randn_like(x_start),
                    torch.randn_like(x_start) * 2)
```

---

### ⭐ Cách 3: Timesteps (T)

Số bước diffusion ảnh hưởng đến chi tiết của process.

#### Vị Trí Code:

```python
diffusion = GaussianDiffusion1D(
    model,
    seq_length=seq_length,
    timesteps=100,  # ← THAY ĐỔI ĐÂY (50, 100, 200, 1000)
    beta_schedule='cosine'
)
```

#### So Sánh:

| T | Training | Sampling | Chất Lượng | Memory |
|---|----------|----------|-----------|--------|
| 50 | 2 min | 1 sec | Tạm | Ít |
| 100 | 5 min | 2 sec | Tốt | Vừa |
| 200 | 10 min | 4 sec | Rất tốt | Nhiều |
| 1000 | 50 min | 20 sec | SOTA | Rất nhiều |

---

### ⭐ Cách 4: Noise Strength Multiplier

Nhân thêm hệ số để tăng/giảm mức độ nhiễu.

#### Vị Trí Code:

```python
# File: GaussianDiffusion1D.q_sample()

NOISE_STRENGTH = 1.0  # ← THAY ĐỔI ĐÂY

return (sqrt_alpha_prod * x_start + 
        sqrt_one_minus_alpha_prod * NOISE_STRENGTH * noise)
```

#### Ảnh Hưởng:

- **0.5:** Noise yếu → Signal giữ lâu hơn → Denoising dễ
- **1.0:** Normal (mặc định)
- **1.5:** Noise mạnh → Noise biến mất nhanh → Denoising khó
- **2.0:** Noise rất mạnh → Nội dung ban đầu mất hết

---

## 📋 Giải Thích Từng Biến

### alphas_cumprod (ᾱ_t)

```python
alphas = 1.0 - betas  # Per-step retention
alphas_cumprod = torch.cumprod(alphas, dim=0)

# ᾱ_t = ∏(1 - β_s) for s=1 to t
# Đại diện cho "phần lượng data gốc giữ lại" sau t bước

# Ví dụ:
# t=0:  ᾱ_0 = 1.0    (100% signal)
# t=50: ᾱ_50 = 0.5   (50% signal, 50% noise)
# t=99: ᾱ_99 = 0.001 (0.1% signal, 99.9% noise)
```

### sqrt_alphas_cumprod

```python
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)

# Hệ số nhân cho signal trong forward diffusion
# x_t = √ᾱ_t * x₀ + ...
```

### sqrt_one_minus_alphas_cumprod

```python
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# Hệ số nhân cho noise trong forward diffusion
# x_t = ... + √(1-ᾱ_t) * ε
```

### posterior_variance

```python
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# Variance của q(x_{t-1}|x_t, x_0) - reverse step variance
# Giúp tính toán bao nhiêu noise cần thêm khi reverse
```

---

## 💡 Ví Dụ Thực Hành

### Ví Dụ 1: Đổi Beta Schedule

```python
# Trước: Cosine
diffusion = GaussianDiffusion1D(
    model,
    seq_length=seq_length,
    timesteps=100,
    beta_schedule='cosine'  # ← Thay đây
)

# Sau: Sigmoid (hiện đại hơn)
diffusion = GaussianDiffusion1D(
    model,
    seq_length=seq_length,
    timesteps=100,
    beta_schedule='sigmoid'
)

# Training lại model...
```

### Ví Dụ 2: Tăng Timesteps để Chất Lượng Cao

```python
# Trước: 100 timesteps (~5 min training)
diffusion = GaussianDiffusion1D(..., timesteps=100)

# Sau: 200 timesteps (~10 min training, nhưng tốt hơn)
diffusion = GaussianDiffusion1D(..., timesteps=200)

# Hoặc thậm chí 1000 (SOTA)
diffusion = GaussianDiffusion1D(..., timesteps=1000)
```

### Ví Dụ 3: Custom Noise Distribution

```python
# Định nghĩa custom diffusion class
class CustomDiffusion1D(GaussianDiffusion1D):
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            # Dùng Uniform thay vì Gaussian
            noise = (torch.rand_like(x_start) * 2 - 1) * math.sqrt(3)
        
        sqrt_alpha_prod = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_prod = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alpha_prod * x_start + sqrt_one_minus_alpha_prod * noise

# Sử dụng
diffusion = CustomDiffusion1D(model, seq_length=seq_length)
```

### Ví Dụ 4: Tăng Noise Strength

```python
# Định nghĩa custom diffusion class
class StrongNoiseDiffusion1D(GaussianDiffusion1D):
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha_prod = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_prod = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        # Nhân 1.5 để noise mạnh hơn
        return sqrt_alpha_prod * x_start + sqrt_one_minus_alpha_prod * 1.5 * noise

# Sử dụng
diffusion = StrongNoiseDiffusion1D(model, seq_length=seq_length)
```

---

## 🎯 Kết Luận

### Tóm Tắt Nơi Thay Đổi Noise:

1. **Beta Schedule** → Kiểm soát mức độ nhiễu ở mỗi bước
2. **Timesteps** → Kiểm soát độ chi tiết của process
3. **Noise Distribution** → Thay đổi loại nhiễu (Gaussian/Uniform/v.v.)
4. **Noise Strength** → Nhân hệ số tăng/giảm mức độ nhiễu

### Khuyến Nghị cho Bắt Đầu:

- ✅ Beta schedule: **'cosine'** (tốt nhất)
- ✅ Timesteps: **100** (balance giữa tốc độ & chất lượng)
- ✅ Noise: **Gaussian** (standard)
- ✅ Strength: **1.0** (normal)

Hãy thử thay đổi **một** variable tại một thời điểm để hiểu rõ ảnh hưởng của nó!
