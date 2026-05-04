This comprehensive cheat sheet summarizes the technical topics and mathematical equations found across your course materials.

---

## **1. Fundamentals & Neural Network Structures**
* [cite_start]**Layer Computation:** The standard feedforward operation is $Z^{(L)} = W^{(L)} \cdot A^{(L)} + b^{(L)}$ [cite: 126] [cite_start]or $Z^{(L)} = W^{(L)} \cdot \phi(Z^{(L-1)}) + b^{(L)}$, where $\phi$ is a non-linear activation function[cite: 127].
* **AdaM Optimization:**
    * [cite_start]**Momentum update:** $M_{k+1} \leftarrow \beta_1 \cdot M_k + (1 - \beta_1) \cdot \nabla J(W_k)$[cite: 200].
    * [cite_start]**Normalizer (second moment):** $V_{k+1} \leftarrow \beta_2 \cdot V_k + (1 - \beta_2) \cdot \nabla J(W_k) \odot \nabla J(W_k)$[cite: 201].
    * [cite_start]**Bias Correction:** $\hat{M}_k \leftarrow \frac{M_k}{(1-[\beta_1]^k)}$ and $\hat{V}_k \leftarrow \frac{V_k}{(1-[\beta_2]^k)}$[cite: 205].
    * [cite_start]**Full Update:** $W_{k+1} \sim W_k - \eta \cdot \frac{\hat{M}_k}{\sqrt{\hat{V}_k + \epsilon}}$[cite: 203].
* **CNN Efficiency Techniques:**
    * **Squeeze Ratio ($SR$):** $SR = \frac{|F_{s1x1}|}{|F_{e1x1}| + [cite_start]|F_{e3x3}|}$, used to control bottlenecking[cite: 386, 387].
    * **Filter Params ($PCT_{3x3}$):** $PCT_{3x3} = \frac{|F_{e3x3}|}{|F_{e1x1}| + [cite_start]|F_{e3x3}|}$, controlling the ratio of $1 \times 1$ vs. $3 \times 3$ filters[cite: 388, 389].

## **2. Transformers & Attention Mechanisms**
* [cite_start]**Vanilla Scaled Dot-Product Attention:** $Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$[cite: 2643, 2709, 3162, 3562].
* [cite_start]**Multi-Head Attention:** $MultiHead(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$[cite: 2644].
* [cite_start]**Layer Normalization:** $LN(E_{tok}) = \gamma_e \left[ \frac{E_{tok} - \mu_{tok}}{\sqrt{\sigma_{tok}^2 + \epsilon}} \right] + \beta_e$, where $\mu$ and $\sigma$ are calculated per token[cite: 2653, 3005].
* [cite_start]**Feed Forward Network (FFN):** $\hat{Z}_{tok} = \sigma(W_{proj} \cdot LN(Z_{tok}) + b_{proj})$[cite: 3005].
* **Linear Attention Efficiency:**
    * [cite_start]**Associative Property:** $(Q \cdot K^T) \cdot V = Q \cdot (K^T \cdot V)$[cite: 3261].
    * **Complexity:** Naive is $O(L^2 \cdot d)$; [cite_start]Efficient/Linear is $O(L \cdot d^2)$ for computation and $O(d^2 + L \cdot d)$ for memory[cite: 3241, 3262].
* **Flash Attention Tiling (Distributed Softmax):**
    * [cite_start]**Local Max:** $m(x) = \max(m(x^{(1)}), m(x^{(2)}))$[cite: 3414].
    * [cite_start]**Scaling factor:** $s_i = e^{m(x^{(i)}) - m(x)}$[cite: 3416].
    * [cite_start]**Global Denominator:** $l(x) = s_1 \cdot l(x^{(1)}) + s_2 \cdot l(x^{(2)})$[cite: 3419, 4816].

## **3. Positional Encoding**
* [cite_start]**Absolute (Sin/Cos):** * $PE_{(p, i \in 0...d_m-1)} = \sin(p/10000^{i/d_m})$[cite: 3569].
    * [cite_start]$PE_{(p, i \in d_m...2d_m)} = \cos(p/10000^{(i-d_m)/d_m})$[cite: 3569].
* **Rotary Positional Encoding (RoPE):**
    * [cite_start]**2D Matrix Rotation:** $R_a \cdot q_a = \begin{bmatrix} \cos(a\theta) & -\sin(a\theta) \\ \sin(a\theta) & \cos(a\theta) \end{bmatrix} \begin{bmatrix} q_1 \\ q_2 \end{bmatrix}$[cite: 3919].
    * [cite_start]**Decoupled Multi-rotation Matrix ($R_a$):** A block-diagonal matrix rotating pairs of features[cite: 3941].
    * [cite_start]**Theta Range:** $\theta_i = 10000^{-2(i-1)/D}$[cite: 3957].

## **4. Ethics & Fairness Metrics**
* **Fairness Definitions:**
    * [cite_start]**Demographic Parity:** $f(X|A=0) \approx f(X|A=1)$[cite: 1458].
    * [cite_start]**Equal Opportunity:** $f(X|A=0, Y=1) \approx f(X|A=1, Y=1)$[cite: 1460].
* **Fairness Loss Functions:**
    * [cite_start]**Counterfactual Loss:** $\mathcal{L}_{cf} = ||f(X_a) - f(X_{a'})||^2$[cite: 1478].
    * [cite_start]**Minimum Difference:** $\mathcal{L}_{md} = \mu(f(X_a)) - \mu(f(X_b))$[cite: 1481].
    * [cite_start]**Total Loss:** $\mathcal{L}_{tot} = \mathcal{L}_{bce} + \lambda \cdot \mathcal{L}_{cf}$[cite: 1479].
* [cite_start]**Retrofitting Embeddings (Implicit De-biasing):** $\Psi(Q) = \sum_{i=1}^n [\alpha_i ||q_i - \hat{q}_i||^2 + \sum_{(i,j)\in E} \beta_{ij} ||q_i - q_j||^2]$, where $\hat{q}$ is the original and $q$ is the new embedding[cite: 1646, 1827].

## **5. Transfer Learning & LLMs**
* [cite_start]**Categories:** * **Inductive:** Same Domain, Different Task ($y_S \neq y_T$)[cite: 2156, 2157].
    * [cite_start]**Transductive:** Different Domains, Same Task ($D_S \neq D_T$)[cite: 2159].
* [cite_start]**BERT Pre-training:** Uses Masked LM and Next Sentence Prediction (NSP)[cite: 4086, 4089].
* [cite_start]**BERT Architecture:** * **Base:** 12 layers, 12 heads, 110M params[cite: 4044, 4045].
    * [cite_start]**Large:** 24 layers, 16 heads, 340M params[cite: 4068, 4069].
* [cite_start]**Reinforcement Learning (RLHF):** $\mathcal{L}_{Total} = \mathcal{L}_2(C) + \lambda \mathcal{L}_1(C)$ used for aligning model responses[cite: 4597].

## **6. State Space Models (SSMs)**
* [cite_start]**Linear Recurrence:** $h_t = A_t \cdot h_{t-1} + B_t \cdot x_t$[cite: 4852].
* [cite_start]**Output Projection:** $y_t = C_t^T \cdot h_t$[cite: 4852].