# Geometric-Inspired Graph-based Incomplete Multi-view Clustering
<!-- A python implement for Geometric-Inspired Graph-based Incomplete Multi-view Clustering -->

Here is the official python implementation of GIGA proposed in ''Geometric-Inspired Graph-based Incomplete Multi-view Clustering'', which is a flexible weight allocation strategy to solve the view missing problem in multi-view clustering.
**Title**: Geometric-Inspired Graph-based Incomplete Multi-view Clustering
**Authors**: Zequn Yang, Han Zhang, Yake Wei, Feiping Nie and Di Hu.
**Paper Resource**: [[Paper]](https://www.sciencedirect.com/science/article/abs/pii/S0031320323007793?dgcid=rss_sd_all)
## Apporach Overview
GIGA addresses the problem of incomplete multi-view clustering. It takes into account the impact of missing views on the weight aggregation strategy which integrate knowledge from different views. Moreover, a geometric-inspired reallocation approach is introduced to mitigate this influence and attain a superior aggregation solution.


 <div align=center><img src="pics/intro.png" width="90%"> 
 
 illustration of our proposed GIGA 
 </div>

<!-- Our method can approach the full-view solution $\bm{s}^*_{(3)}$ using partila view, and obtain the optimal solution $\bm{s}^*_{(2)}$ which has the maximum cosine similarity with the full-view solution. -->

Our method can approximate the full-view solution, denoted as $s^*_{(3)}$, using just available views. We obtain the optimal solution $s^*_{(2)}$ on available views through projection, which has the highest cosine similarity to the full-view solution.

 <div align=center><img src="pics/method.png" width="80%"> 
 
 Detailed illustration of our geometric analysis 
 </div>

## Get Started
<pre><code>
pip install -r requirements.txt 
python main_GIGA.py
</code></pre>

## Data Preparation

The data should be located in the path ./data/, saved as .npy files. For each data containing $l$ views $X^1, X^2...,X^l$, the data should be arranged as the list $[X^1, X^2...,X^l]$, where $X^v \in \mathbb{R}^{d_v \times n}$, where $d_v$ is the dimension of the data in $v$-th view. Datasets can be found in [[link]](https://github.com/wangsiwei2010/awesome-multi-view-clustering#jump3).

## Citation

If you find this work useful, please consider citing it.

<pre><code>
@article{yang2023geometric,
  title={Geometric-Inspired Graph-based Incomplete Multi-view Clustering},
  author={Yang, Zequn and Zhang, Han and Wei, Yake and Wang, Zheng and Nie, Feiping and Hu, Di},
  journal={Pattern Recognition},
  pages={110082},
  year={2023},
  publisher={Elsevier}
}
</code></pre>

## Acknowledgement

This research was supported by Public Computing Cloud, Renmin University of China.