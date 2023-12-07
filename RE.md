**2. representation enhancement strategy**

We conducted additional experiments and provided a theoretical analysis showcasing how the representation enhancement strategy contributes to learning information from a modality riddled with noise. Our focus was on the Packet Size and TTL modalities for the SJTU-AN21 dataset, with the introduction of 50% noise into the Packet Size modality. The table presents experimental results, emphasizing the superior efficacy of the RE method compared to the other fusion techniques. Furthermore, our analysis revealed two crucial findings:
- The training loss for all three fusion methods is exceptionally low, indicating that the fused predictions closely align with the corresponding label values across the entire training dataset.
- In the noise-filled Packet Size modality, the Add and Concatenation fusion methods demonstrate subpar performance.

This indicates that the Add and Concatenation fusion methods overly depend on the TTL modality, neglecting the opportunity to glean information from the Packet Size modality. In contrast, the experimental results of the RE method demonstrate its ability to extract valuable information from the noisy Packet Size modality, achieving an accuracy of 67.79%.

| Fusion Method | Training Loss (Length) | Training Loss (TTL) | Training Loss (Fusion) | AC (Length) | AC (TTL)   | AC (Fusion) | 
| ------------- | ---------------------- | ------------------- | ---------------------- | ----------- | ---------- | ----------- |
| Add           | 2.60                   | 0.32                | 0.05                   | 0.3511      | 0.7576     | 0.8826      |
| Concatenation | 2.69                   | 0.31                | 0.05                   | 0.2148      | 0.7209     | 0.8394      |
| RE            | **0.67**               | **0.13**            | **0.04**               | **0.6779**  | **0.7993** | **0.8948**  |

We provide a theoretical analysis of the above phenomena to explain why the Add and Concatenation fusion methods neglect to learn information from the noisy Packet Size modality.

**Preliminaries:**
Given the fused representation $\mathbf{Z}=[z_1,z_2,\cdots,z_D]$, where $D$ is the representation dimension. We use a fully-connected layer to compute the output logit:

$$
\mathbf{H}=[h_1,h_2,\cdots,h_K]=\mathbf{Z}\mathbf{W}^T=
\left[\begin{array}{c}
z_1 & z_2 & \cdots & z_D \\
\end{array}\right]\times
\left[\begin{array}{c}
w_{11} & w_{12} & \cdots & w_{1D}\\
w_{21} & w_{22} & \cdots & w_{2D}\\
\vdots & \vdots & \ddots & \vdots\\
w_{K1} & w_{K2} & \cdots & w_{KD}\\
\end{array}\right]^T
,
$$

where $\mathbf{W}\in\mathbb{R}^{K\times D}$ is the weight matrix of the fully-connected layer and $K$ is the number of classes. According to the above formula, We have:

$$
h_i=\sum_{j=1}^Dz_jw_{ij}.
$$

Then the softmax output can be calculated by:

$$
\mathbf{Q}=[q_1,q_2,\cdots,q_K]=softmax(\mathbf{H})=softmax([h_1,h_2,\cdots,h_K]).
$$

According to the softmax formula, we have:

$$
q_i=\frac{e^{h_i}}{\sum\limits _{j=1}^Ke^{h_j}}.
$$

The softmax gradient is well known as follows:

$$
\frac{\partial q_j}{\partial h_i}=
\begin{aligned}
q_j(1-q_j),i=j\\
-q_iq_j,i\neq j
\end{aligned}
$$

The final classification loss is calculated by cross-entropy (assume $k$ is the ground-truth label):

$$
\mathcal{L}=-log(q_k).
$$

**Prove:**
We first solve for the gradient of $\mathcal{L}$ with respect to $h_i$ through the chain rule:

$$
\frac{\partial\mathcal{L}}{\partial h_i}=\sum_{j=1}^K\frac{\partial\mathcal{L}}{\partial q_j}\cdot\frac{\partial q_j}{\partial h_i}=-\frac{1}{q_k}\cdot\frac{\partial q_k}{\partial h_i}=
\begin{aligned}
-\frac{1}{q_k}\cdot q_k(1-q_k)=q_k-1,i=k\\
-\frac{1}{q_k}\cdot -q_iq_k=q_i,i\neq k
\end{aligned}
$$

Then we calcuate the gradient of $\mathcal{L}$ with respect to representation $z_i$:

$$
\begin{gather*}
\frac{\partial\mathcal{L}}{\partial z_i}=\sum_{j=1}^K\frac{\partial\mathcal{L}}{\partial h_j}\cdot \frac{\partial h_j}{\partial z_i}=\frac{\partial\mathcal{L}}{\partial h_k}\cdot \frac{\partial h_k}{\partial z_i}+\sum_{j=1, j\neq k}^K\frac{\partial\mathcal{L}}{\partial h_j}\cdot \frac{\partial h_j}{\partial z_i}\\
=(q_k-1)\cdot w_{ki}+\sum_{j=1, j\neq k}^Kq_j\cdot w_{ji}=\sum_{j=1}^Kq_jw_{ji}-w_{ki}.
\end{gather*}
$$

As mentioned in the above analysis, the training loss of the three fusion methods is very low, which means that the fusion predictions of the model are close to the ground-truth label, that is:

$$
q_i \to
\begin{aligned}
1,i=k\\
0,i\neq k
\end{aligned}
$$

Finally, we have:

$$
\frac{\partial\mathcal{L}}{\partial z_i}\to 0.
$$

The preceding analysis indicates that when the model acquires sufficient knowledge from a specific modality to generate high-confidence predictions, it tends to disregard learning from other modalities, particularly those with noise. Conversely, the representation enhancement strategy compels the model to glean as much knowledge as feasible from each modality by introducing random adjustments to the weights of individual modalities throughout the training phase. This ensures that the model can effectively learn valuable information even in the presence of noisy modalities. We plan to incorporate this segment, including additional experimental results and analyses, into the final version.
