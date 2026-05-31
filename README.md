<div dir="ltr" align="left">

<p align="center">
  <img src="logos/macquarie_logo.png" alt="Macquarie University" width="230"/>
</p>

<h1 align="center">TempEdge-ResGNN for Bitcoin Illicit Transaction Detection</h1>

<p align="center">
  <strong>COMP8221 — Advanced Machine Learning</strong><br/>
  <strong>Assignment 2 — Project Option 1: Real-world Applications of GNNs</strong>
</p>

<p align="center">
  A PyTorch Geometric project for detecting illicit Bitcoin transactions on the Elliptic transaction graph,<br/>
  using classical baselines, standard GNNs, and a custom edge-aware residual GNN.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-blue">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.11-ee4c2c">
  <img alt="PyG" src="https://img.shields.io/badge/PyG-2.7.0-3b82f6">
  <img alt="Dataset" src="https://img.shields.io/badge/Dataset-Elliptic%20Bitcoin-brightgreen">
  <img alt="Task" src="https://img.shields.io/badge/Task-Node%20Classification-purple">
</p>

<hr/>

<h2>Overview</h2>

<p>
This repository contains my COMP8221 Assignment 2 project on <strong>Bitcoin illicit transaction detection</strong>
using the <strong>Elliptic Bitcoin transaction graph</strong>.
</p>

<p>
The task is framed as a <strong>semi-supervised node-classification problem</strong>. Each node is a Bitcoin transaction,
each directed edge is a payment-flow link, and labelled nodes are classified as <strong>licit</strong> or
<strong>illicit</strong>. Unknown-label nodes are kept in the graph for message passing but excluded from the supervised
loss and evaluation.
</p>

<p>
The main point of the project is not just to run a standard GNN. I compare feature-based models, neural baselines,
standard GNNs, and my custom model, <strong>TempEdge-ResGNN</strong>, then analyse where the graph models help and where
they still struggle.
</p>

<p>
The final result is honest: <strong>Random Forest is the strongest overall model</strong>, while
<strong>TempEdge-ResGNN is the strongest neural/GNN model</strong> in my experiments.
</p>

<hr/>

<h2>Project Snapshot</h2>

<table dir="ltr">
  <tr><th align="left">Item</th><th align="left">Details</th></tr>
  <tr><td><strong>Student</strong></td><td>Alireza Yegane</td></tr>
  <tr><td><strong>Student ID</strong></td><td>60957107</td></tr>
  <tr><td><strong>Unit</strong></td><td>COMP8221 — Advanced Machine Learning</td></tr>
  <tr><td><strong>Assignment</strong></td><td>Assignment 2</td></tr>
  <tr><td><strong>Project option</strong></td><td>Option 1 — Real-world applications of GNNs</td></tr>
  <tr><td><strong>Dataset</strong></td><td>Elliptic Bitcoin transaction graph</td></tr>
  <tr><td><strong>Task</strong></td><td>Semi-supervised binary node classification</td></tr>
  <tr><td><strong>Framework</strong></td><td>PyTorch Geometric</td></tr>
  <tr><td><strong>Main notebook</strong></td><td><code>COMP8221_TempEdgeResGNN_v2.ipynb</code></td></tr>
  <tr><td><strong>Executed notebook</strong></td><td><code>COMP8221_TempEdgeResGNN_v2_executed_for_review.ipynb</code></td></tr>
  <tr><td><strong>Final report</strong></td><td><code>2026S1 COMP8221 Assignment 2 60957107 Alireza_Yegane.pdf</code></td></tr>
</table>

<hr/>

<h2>Results at a Glance</h2>

<table dir="ltr">
  <tr><th align="left">Category</th><th align="left">Model</th><th align="left">Key result</th></tr>
  <tr>
    <td><strong>Best overall</strong></td>
    <td>Random Forest</td>
    <td>Illicit-F1 <strong>0.8058</strong>, PR-AUC <strong>0.7820</strong></td>
  </tr>
  <tr>
    <td><strong>Best neural / GNN</strong></td>
    <td>TempEdge-ResGNN</td>
    <td>Illicit-F1 <strong>0.5760 ± 0.0672</strong>, PR-AUC <strong>0.6233 ± 0.0516</strong></td>
  </tr>
  <tr>
    <td><strong>Main challenge</strong></td>
    <td>Temporal distribution shift</td>
    <td>Sharp performance drop after timestep 43</td>
  </tr>
</table>

<p>
The results show that GNNs improve over a plain GCN on this dataset, but they do <strong>not</strong> automatically beat
a strong feature-based baseline. This is an important part of the analysis because the Elliptic dataset already contains
strong handcrafted transaction features.
</p>

<hr/>

<h2>Why This Project Fits Option 1</h2>

<p>
This project is a real-world GNN application because the graph structure is part of the problem, not something added
artificially. Bitcoin transaction monitoring depends on how transactions are connected. A suspicious transaction may not
look obvious by itself, but its surrounding payment-flow pattern can provide useful context.
</p>

<table dir="ltr">
  <tr><th align="left">Requirement</th><th align="left">Evidence in this submission</th></tr>
  <tr><td><strong>Real-world use case</strong></td><td>Bitcoin anti-money-laundering transaction monitoring</td></tr>
  <tr><td><strong>Public graph dataset</strong></td><td>Elliptic Bitcoin transaction graph through PyTorch Geometric</td></tr>
  <tr><td><strong>Graph ML formulation</strong></td><td>Semi-supervised binary node classification</td></tr>
  <tr><td><strong>Multiple model comparisons</strong></td><td>Logistic Regression, Random Forest, MLP, GCN, GraphSAGE, GAT, GATv2, TempEdge-ResGNN</td></tr>
  <tr><td><strong>Custom model design</strong></td><td>Edge-aware residual GNN with timestep features and gating</td></tr>
  <tr><td><strong>Ablation study</strong></td><td>Edge attributes, residual gate, time features, JKNet, focal loss, old timestep embedding</td></tr>
  <tr><td><strong>Analysis beyond headline metrics</strong></td><td>PR curves, confusion matrices, per-timestep F1, temporal shift breakdown</td></tr>
</table>

<hr/>

<h2>Method Summary</h2>

<h3>1. Dataset and Task Setup</h3>

<p>
The Elliptic graph contains Bitcoin transactions as nodes and payment-flow links as directed edges. The labelled nodes
are classified as either <strong>licit</strong> or <strong>illicit</strong>, while unknown-labelled nodes remain in the graph for
message passing.
</p>

<h3>2. Temporal Split</h3>

<p>
I use a temporal split rather than a random split. This is more realistic for AML because a real system is trained on
past transactions and used on future transactions. Validation nodes are carved from timesteps <strong>30–34</strong>, and the
test period uses later timesteps.
</p>

<h3>3. Preprocessing</h3>

<p>
The 165 numerical node features are standardised using <code>StandardScaler</code>, fitted only on training nodes. This avoids
leaking future-period statistics into preprocessing. Unknown labels are excluded from loss and metrics instead of being
treated as licit.
</p>

<h3>4. Baselines</h3>

<ul>
  <li><strong>Logistic Regression</strong> — simple feature-based reference</li>
  <li><strong>Random Forest</strong> — strong feature-based baseline</li>
  <li><strong>MLP</strong> — feature-only neural model</li>
  <li><strong>GCN</strong> — standard graph convolution baseline</li>
  <li><strong>GraphSAGE</strong> — neighbourhood aggregation baseline</li>
  <li><strong>GAT / GATv2</strong> — attention-based GNN baselines</li>
</ul>

<p>
Including Random Forest is important because it tests whether graph neural networks actually add value beyond the
handcrafted Elliptic features.
</p>

<h3>5. TempEdge-ResGNN</h3>

<p>
The custom model combines edge-aware message passing, residual gating, non-parametric timestep features, and optional
Jumping Knowledge aggregation. The timestep features are deliberately non-parametric because the model must generalise
to later timesteps.
</p>

<hr/>

<h2>Important Findings</h2>

<h3>Random Forest is strongest overall</h3>

<table dir="ltr">
  <tr><th align="left">Metric</th><th align="right">Value</th></tr>
  <tr><td>Illicit precision</td><td align="right"><strong>0.9175</strong></td></tr>
  <tr><td>Illicit recall</td><td align="right"><strong>0.7184</strong></td></tr>
  <tr><td>Illicit-F1</td><td align="right"><strong>0.8058</strong></td></tr>
  <tr><td>ROC-AUC</td><td align="right"><strong>0.9115</strong></td></tr>
  <tr><td>PR-AUC</td><td align="right"><strong>0.7820</strong></td></tr>
</table>

<p>
This does not invalidate the GNN work. It shows that the Elliptic handcrafted features are very strong, and that GNNs
should be compared against strong baselines rather than only weak neural models.
</p>

<h3>TempEdge-ResGNN is the best neural/GNN model</h3>

<table dir="ltr">
  <tr><th align="left">Metric</th><th align="right">Value</th></tr>
  <tr><td>Illicit-F1</td><td align="right"><strong>0.5760 ± 0.0672</strong></td></tr>
  <tr><td>PR-AUC</td><td align="right"><strong>0.6233 ± 0.0516</strong></td></tr>
  <tr><td>ROC-AUC</td><td align="right"><strong>0.8922 ± 0.0069</strong></td></tr>
</table>

<h3>Temporal shift is the main difficulty</h3>

<p>
The per-timestep analysis shows that the model performs much better before timestep 43 than after it. This suggests that
the hardest part of the problem is not only architecture design, but also temporal distribution shift.
</p>

<hr/>

<h2>Ablation Summary</h2>

<table dir="ltr">
  <tr><th align="left">Ablation</th><th align="right">Illicit-F1</th></tr>
  <tr><td>Full model</td><td align="right"><strong>0.5838 ± 0.0440</strong></td></tr>
  <tr><td>No edge attributes</td><td align="right">0.5597 ± 0.0621</td></tr>
  <tr><td>No residual gate</td><td align="right">0.5630 ± 0.0423</td></tr>
  <tr><td>No time features</td><td align="right">0.5290 ± 0.0323</td></tr>
  <tr><td>No JKNet</td><td align="right">0.5861 ± 0.0464</td></tr>
  <tr><td>Class-balanced focal loss</td><td align="right">0.5606 ± 0.0478</td></tr>
  <tr><td>Old learned timestep embedding</td><td align="right">0.3351 ± 0.0407</td></tr>
</table>

<p>
The strongest lesson is that non-parametric time features are much safer than the old learned timestep embedding.
JKNet was not clearly beneficial, so I do not claim it as a major improvement.
</p>

<hr/>

<h2>Repository Structure</h2>

<pre dir="ltr"><code>.
├── COMP8221_TempEdgeResGNN_v2.ipynb
├── COMP8221_TempEdgeResGNN_v2_executed_for_review.ipynb
├── README.md
├── 2026S1 COMP8221 Assignment 2 60957107 Alireza_Yegane.pdf
├── results/
│   ├── all_model_results.csv
│   ├── main_results_summary.csv
│   ├── ablation_results.csv
│   ├── per_timestep_metrics.csv
│   └── distribution_shift_breakdown.csv
└── figures/
    ├── class_distribution.png
    ├── timestep_distribution.png
    ├── degree_distribution.png
    ├── graph_visualisation.png
    ├── tempedge_resgnn_architecture.png
    ├── model_comparison_f1.png
    ├── model_comparison_prauc.png
    ├── learning_curves.png
    ├── pr_curves.png
    ├── confusion_matrix_best_model.png
    ├── confusion_matrix_grid.png
    ├── per_timestep_f1.png
    └── ablation_illicit_f1.png</code></pre>

<p>
The Elliptic dataset is downloaded automatically by PyTorch Geometric on first run and is not stored in this repository.
</p>

<hr/>

<h2>How to Run</h2>

<h3>Environment</h3>

<table dir="ltr">
  <tr><th align="left">Package</th><th align="left">Version</th></tr>
  <tr><td>Python</td><td>3.12.3</td></tr>
  <tr><td>PyTorch</td><td>2.11.0+cu128</td></tr>
  <tr><td>PyTorch Geometric</td><td>2.7.0</td></tr>
  <tr><td>scikit-learn</td><td>latest compatible</td></tr>
  <tr><td>pandas / numpy / matplotlib / networkx</td><td>latest compatible</td></tr>
</table>

<p>Install the main dependencies with:</p>

<pre dir="ltr"><code>pip install torch torch_geometric scikit-learn pandas numpy matplotlib networkx</code></pre>

<h3>Main usage</h3>

<pre dir="ltr"><code>jupyter notebook COMP8221_TempEdgeResGNN_v2.ipynb</code></pre>

<p>Then run all cells in order. The important configuration is near the start of the notebook:</p>

<pre dir="ltr"><code>CONFIG["MODE"] = "quick"   # or "full"</code></pre>

<ul>
  <li><code>quick</code> is for a smoke test.</li>
  <li><code>full</code> reproduces the submitted Version 2 results.</li>
</ul>

<hr/>

<h2>Generated Outputs</h2>

<table dir="ltr">
  <tr><th align="left">File</th><th align="left">Description</th></tr>
  <tr><td><code>all_model_results.csv</code></td><td>One row per model/seed run</td></tr>
  <tr><td><code>main_results_summary.csv</code></td><td>Main model-comparison table</td></tr>
  <tr><td><code>ablation_results.csv</code></td><td>Ablation runs</td></tr>
  <tr><td><code>per_timestep_metrics.csv</code></td><td>Per-timestep illicit-F1</td></tr>
  <tr><td><code>distribution_shift_breakdown.csv</code></td><td>Pre/post timestep 43 breakdown</td></tr>
</table>

<table dir="ltr">
  <tr><th align="left">Figure group</th><th align="left">Files</th></tr>
  <tr><td>Data / task</td><td><code>class_distribution</code>, <code>timestep_distribution</code>, <code>degree_distribution</code>, <code>graph_visualisation</code></td></tr>
  <tr><td>Model</td><td><code>tempedge_resgnn_architecture</code></td></tr>
  <tr><td>Evaluation</td><td><code>model_comparison_f1</code>, <code>model_comparison_prauc</code>, <code>learning_curves</code>, <code>pr_curves</code>, <code>confusion_matrix_best_model</code>, <code>confusion_matrix_grid</code></td></tr>
  <tr><td>Analysis</td><td><code>per_timestep_f1</code>, <code>ablation_illicit_f1</code></td></tr>
</table>

<hr/>

<h2>Reproducibility Notes</h2>

<ul>
  <li>Python, NumPy, and PyTorch seeds are fixed.</li>
  <li>Neural models are run across <strong>five seeds</strong> in the full Version 2 setting.</li>
  <li>Threshold tuning is done only on the validation set.</li>
  <li>The chosen threshold is then applied once to the test set.</li>
  <li>Feature scaling is fitted on training nodes only.</li>
  <li>Unknown-labelled nodes are kept for message passing but excluded from loss and metrics.</li>
  <li>Small GPU-level variation is still possible because some PyTorch Geometric scatter operations are non-deterministic.</li>
</ul>

<hr/>

<h2>References</h2>

<pre dir="ltr"><code>@inproceedings{weber2019amlbitcoin,
  title     = {Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics},
  author    = {Weber, Mark and Domeniconi, Giacomo and Chen, Jie and Weidele, Daniel Karl I. and Bellei, Claudio and Robinson, Tom and Leiserson, Charles E.},
  booktitle = {KDD Workshop on Anomaly Detection in Finance},
  year      = {2019}
}

@inproceedings{kipf2017gcn,
  title     = {Semi-Supervised Classification with Graph Convolutional Networks},
  author    = {Kipf, Thomas N. and Welling, Max},
  booktitle = {ICLR},
  year      = {2017}
}

@inproceedings{hamilton2017graphsage,
  title     = {Inductive Representation Learning on Large Graphs},
  author    = {Hamilton, William L. and Ying, Rex and Leskovec, Jure},
  booktitle = {NeurIPS},
  year      = {2017}
}

@inproceedings{velickovic2018gat,
  title     = {Graph Attention Networks},
  author    = {Veličković, Petar and others},
  booktitle = {ICLR},
  year      = {2018}
}

@inproceedings{brody2022gatv2,
  title     = {How Attentive are Graph Attention Networks?},
  author    = {Brody, Shaked and Alon, Uri and Yahav, Eran},
  booktitle = {ICLR},
  year      = {2022}
}

@inproceedings{xu2018jknet,
  title     = {Representation Learning on Graphs with Jumping Knowledge Networks},
  author    = {Xu, Keyulu and Li, Chengtao and Tian, Yonglong and Sonobe, Tomohiro and Kawarabayashi, Ken-ichi and Jegelka, Stefanie},
  booktitle = {ICML},
  year      = {2018}
}

@inproceedings{cui2019classbalanced,
  title     = {Class-Balanced Loss Based on Effective Number of Samples},
  author    = {Cui, Yin and Jia, Menglin and Lin, Tsung-Yi and Song, Yang and Belongie, Serge},
  booktitle = {CVPR},
  year      = {2019}
}</code></pre>

<hr/>

<p align="center">
  <strong>Macquarie University</strong><br/>
  COMP8221 · Advanced Machine Learning · 2026 S1
</p>

</div>
