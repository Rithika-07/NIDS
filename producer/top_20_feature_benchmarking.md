The results from the previous benchmarking run (obtained before the process was stopped) are shown below:

### 1. Selected Top 20 Features (Ranked by Importance)
*   **Numeric (19):** `ct_state_ttl` (0.164), `sttl` (0.094), `dttl` (0.051), `sload` (0.050), `rate` (0.046), `dload` (0.044), `tcprtt` (0.041), `dmean` (0.039), `sbytes` (0.036), `dinpkt` (0.031), `ackdat` (0.028), `sinpkt` (0.027), `synack` (0.027), `ct_srv_dst` (0.026), `dur` (0.023), `dpkts` (0.023), `smean` (0.023), `ct_dst_src_ltm` (0.022), `ct_srv_src` (0.021)
*   **Categorical (1):** `proto` (0.029)

---

### 2. Multi-Class Model Comparison Metrics
Here is the performance of each evaluated pipeline configuration on the multi-class (`attack_cat`) classification:

| Model & Encoder Configuration | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Avg FPR | Train Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest (Balanced) [target encoder]** | **80.76%** | **0.6224** | **0.5809** | **0.5917** | **0.0221** | **2.40s** |
| Random Forest (Unweighted) [target encoder] | 82.53% | 0.7740 | 0.5420 | 0.5728 | 0.0211 | 2.25s |
| SMOTE + Random Forest [onehot encoder] | 79.51% | 0.5721 | 0.6128 | 0.5835 | 0.0229 | 12.08s |
| SMOTE + Random Forest [target encoder] | 75.63% | 0.5407 | 0.6302 | 0.5538 | 0.0262 | 17.51s |
| SMOTE + HistGradientBoosting [onehot encoder] | 78.58% | 0.5633 | 0.6451 | 0.5776 | 0.0235 | 50.03s |
| HistGradientBoosting (Balanced) [target encoder] | 74.79% | 0.5653 | 0.6790 | 0.5683 | 0.0267 | 3.07s |
| Random Forest (Unweighted) [onehot encoder] | 81.93% | 0.7190 | 0.5307 | 0.5620 | 0.0218 | 1.79s |
| HistGradientBoosting (Balanced) [onehot encoder] | 74.54% | 0.5558 | 0.6662 | 0.5592 | 0.0269 | 8.31s |
| Random Forest (Balanced) [onehot encoder] | 77.80% | 0.5919 | 0.5386 | 0.5464 | 0.0246 | 1.79s |
| SMOTE + HistGradientBoosting [target encoder] | 73.94% | 0.5313 | 0.6617 | 0.5358 | 0.0276 | 13.82s |
| **Baseline Random Forest (Original 13 features)** | **81.52%** | **0.6970** | **0.4999** | **0.5294** | **0.0226** | **1.99s** |
| Balanced Random Forest [target encoder] | 71.28% | 0.4963 | 0.6558 | 0.4816 | 0.0304 | 0.96s |
| Balanced Random Forest [onehot encoder] | 71.09% | 0.4916 | 0.6315 | 0.4719 | 0.0306 | 0.62s |

#### **Winner:** `Random Forest (Balanced) [target encoder]`
*   **Macro F1-Score:** **0.5917** (A significant improvement over the Baseline model's **0.5294**).
*   **Minority Recall:** Handled the class imbalance natively without requiring the computationally heavy SMOTE process.

---

### 3. Binary Classification Model Results
*   **Binary Model Accuracy:** **95.59%**
*   **Binary Model F1-Score:** **0.9680**