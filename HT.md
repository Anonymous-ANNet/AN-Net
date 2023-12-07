**1. high temperature self-attention mechanism**

We provide a theoretical analysis demonstrating how high temperature self-attention mechanism produces extremely small weights for short-term features of irrelevant flows. As stated in Equation (5), normal self-attention layer scales dot products $\mathbf{S}_i$ by $\frac{1}{\sqrt{D}}$ before applying a softmax function (for simplicity, we omit the subscript $i$):

$$
\mathbf{W}=softmax(\frac{\mathbf{S}}{\sqrt{D}})=[w_{1},w_{2},\cdots,w_{N}],\sum^N_{j=1}w_{j}=1.
$$

As a comparison, high temperature self-attention mechanism increases the magnitude of $\mathbf{S}$ through the temperature hyper-parameter $\tau$. 
The vector $\mathbf{S}$ can be decomposed into two components:

$$
\mathbf{S}=||\mathbf{S}||\cdot\hat{\mathbf{S}},
$$

where $||\mathbf{S}||=\sqrt{s_1^2+s_2^2+\cdots+s_N^2}$ is the Euclidean norm, and $\hat{\mathbf{S}}=[\hat{s}_1,\hat{s}_2,\cdots,\hat{s}_N]$ is the unit vector in the same direction as $\mathbf{S}$. In other word, $||\mathbf{S}||$ and $\hat{\mathbf{S}}$ represent the *magnitude* and the *direction* of $\mathbf{S}$, respectively.
Assume that the dot product calculated from the irrelevant flow is relatively small: $\hat{s}_k=\mathop{min}\limits _j(\hat{s}_j)$, then the weight for the irrelevant flow $w_k$ can be calculated according to the softmax formula:

$$
w_k=\frac{e^{||\mathbf{S}||\hat{s}_k}}{\sum\limits _{j=1}^Ne^{||\mathbf{S}||\hat{s}_j}}.
$$

High temperature self-attention mechanism increases the magnitude of $||\mathbf{S}||$ through the temperature hyper-parameter $\tau$ and the weight can be expressed as:

$$
w_k'=\frac{e^{\frac{||\mathbf{S}||}{\tau}\hat{s}_k}}{\sum\limits _{j=1}^Ne^{\frac{||\mathbf{S}||}{\tau}\hat{s}_j}},
$$

where $\tau\ll 1$ is the temperature hyper-parameter.
Let $t=||\mathbf{S}||\cdot(\frac{1}{\tau}-1)>0$, then we have

$$
w_k'=\frac{e^{(||\mathbf{S}||+t)\hat{s}_k}}{\sum\limits _{j=1}^Ne^{(||\mathbf{S}||+t)\hat{s}_j}}=\frac{e^{||\mathbf{S}||\hat{s}_k}}{\sum\limits _{j=1}^Ne^{||\mathbf{S}||\hat{s}_j+t(\hat{s}_j-\hat{s}_k)}}.
$$

For any $j\in[1,2,\cdots,N]$, we have $\hat{s}_j-\hat{s}_k\ge 0$. Then

$$
w_k'\le w_k.
$$

In conclusion, increasing the magnitude $||\mathbf{S}||$ will cause a sharp distribution for weight score $\mathbf{W}$. By producing extremely small weights for short-term features from irrelevant flows, high temperature self-attention mechanism can effectively resist irrelevant packet noise.
