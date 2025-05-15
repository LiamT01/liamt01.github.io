---
layout: distill
title: "Scaling Like a Pro: Zero Bubble Pipeline Parallelism Demystified"
description:
  Pipeline parallelism is key to efficient distributed training for large-scale models, but its performance is often hindered by pipeline bubbles, which are gaps in computation that limit throughput.
  A recent paper introduces a breakthrough zero-bubble scheduling strategy, achieving up to 30% throughput improvement.
  In this post, we demystify the scheduling process with detailed, step-by-step illustrations, providing clarity and context that complement the original work.
  Whether you're new to ML systems or a seasoned researcher, this post bridges the gap between high-level concepts and practical understanding with fresh and accessible perspectives.
tags: [distributed-training, scaling]
categories: [machine-learning]
giscus_comments: true
related_posts: true
date: 2024-11-26
htmlwidgets: true
hidden: false

authors:
  - name: Hongwei Tu
    affiliations:
      name: SCS, CMU
  - name: Jingwei Dai
    url: "https://www.heinz.cmu.edu/faculty-research/profiles/dai-jingwei/"
    affiliations:
      name: Heinz College, CMU

# must be the exact same name as your blogpost
bibliography: "2024-11-26-zero-bubble.bib"

# Add a table of contents to your post.
toc:
  - name: Introduction
  - name: Background
    subsections:
      - name: Distributed Training and Parallelism
      - name: Pipeline Parallelism
  - name: The 1F1B Approach
  - name: Toward Zero Bubbles
    subsections:
      - name: "Handcrafted Schedule 1: ZB-H1"
      - name: "Handcrafted Schedule 2: ZB-H2"
  - name: Automatic Zero-Bubble Scheduling
    subsections:
      - name: From Heuristics to Automation
  - name: Experiments and Results
    subsections:
      - name: Key Findings
  - name: Limitations
  - name: Conclusion
    subsections:
      - name: Contributions
      - name: Future Work
---

## Introduction

Distributed training has become indispensable to deep learning, enabling researchers to scale models to billions and even trillions of parameters.
Among the key strategies for distributing workloads across multiple GPUs is **pipeline parallelism**, a technique that splits a model into stages, with each stage assigned to a different device.
Like an assembly line in manufacturing, pipeline parallelism allows multiple stages of a model to be processed simultaneously, improving throughput and making large-scale training feasible.

However, pipeline parallelism is not without challenges.
A major inefficiency stems from **pipeline bubbles**, which occur when devices are idle due to sequential dependencies between computation stages.
These idle periods limit throughput and waste computational resources, particularly during the warm-up and flush phases of a pipeline.
Over the years, several scheduling strategies, such as **1F1B** (one-forward-one-backward) <d-cite key="harlap2018pipedream, fan2021dapple, narayanan2021efficient"/>, have been developed to mitigate bubbles, but none have entirely eliminated them—until now.

A recent paper, "Zero Bubble (Almost) Pipeline Parallelism" <d-cite key="qi2024zero"/>, introduces a breakthrough approach that achieves **zero pipeline bubbles** under synchronous training semantics.
By splitting the backward pass into two parts: gradients with respect to **inputs** and gradients with respect to **weights**, the authors strategically place operations to fill bubbles and propose an **automatic scheduling algorithm**.

In this post, we demystify the scheduling process, starting with the commonly used 1F1B approach and progressing to the zero-bubble schedules.
Through **step-by-step derivations and visualizations**, we show how these schedules are constructed and highlight their key advantages.
Finally, we explore how the insights from these schedules are generalized into an automatic scheduling algorithm.
Whether you're a seasoned ML systems researcher or new to distributed training, this post will provide clarity and context to help you understand the exciting potential of zero-bubble pipeline parallelism.

## Background

### Distributed Training and Parallelism

As deep learning models grow larger and more complex, training them on a single GPU becomes infeasible due to memory and compute constraints.
Distributed training solves this problem by splitting the workload across multiple devices.
The two most widely used parallelism strategies are **data parallelism** <d-cite key="goyal2017accurate, li2020pytorch"/> and **model parallelism** <d-cite key="harlap2018pipedream, huang2019gpipe, fan2021dapple, zheng2022alpa"/>.

#### Data Parallelism

In data parallelism, the dataset is divided into smaller mini-batches, and identical copies of the model are deployed across multiple devices.
Each device processes a different mini-batch, computes gradients, and synchronizes the results through an all-reduce operation.

{% include figure.liquid loading="eager" path="assets/img/2024-11-26-zero-bubble/data_parallelism.svg" class="img-fluid rounded z-depth-1 p-2" zoomable=true caption="Data parallelism across four devices." %}

While simple and effective for small-to-medium-sized models, data parallelism struggles with memory-intensive models because it requires each device to store a complete copy of the model.
In addition, the cost of communicating gradients grows with the number of devices, creating a bottleneck that can slow down training.

#### Model Parallelism

In model parallelism, the model itself is split across devices, with each device responsible for a subset of the computations.
Two common forms of model parallelism are:

- **Tensor parallelism**: Splits tensors (e.g., weight matrices) within an operation across devices.
- **Pipeline parallelism**: Divides the model into stages (layers), with each stage assigned to a device. Data flows through the stages sequentially, similar to an assembly line.

### Pipeline Parallelism

{% include figure.liquid loading="eager" path="assets/img/2024-11-26-zero-bubble/pipeline_parallelism.svg" class="img-fluid rounded z-depth-1 p-2" zoomable=true caption="Pipeline parallelism across four devices." %}

Pipeline parallelism is particularly useful for training large models that exceed the memory capacity of a single GPU.
By splitting the model into **stages** and the data into **micro-batches**, it enables simultaneous execution of forward and backward passes on different micro-batches and model layers.
For example, while the first stage processes the forward pass for one micro-batch, the second stage can work on the backward pass for another micro-batch.

However, pipeline parallelism introduces **pipeline bubbles**, which are idle periods that occur because:

1. Stages must wait for data from preceding stages (sequential dependency).
2. The warm-up and flush phases (described later) inherently result in under-utilized devices.

{% include figure.liquid loading="eager" path="assets/img/2024-11-26-zero-bubble/bubbles.svg" class="img-fluid rounded z-depth-1 p-2" zoomable=true caption="Bubbles in pipeline parallelism. Source: <a href='https://proceedings.neurips.cc/paper_files/paper/2019/file/093f65e080a295f8076b1c5722a46aa2-Paper.pdf'>Gpipe</a>." %}

## The 1F1B Approach

To mitigate bubbles, practitioners have employed different **scheduling strategies**.
One widely adopted strategy is **1F1B** (one-forward-one-backward), which alternates between forward and backward passes to balance workloads across devices.

Before we dive into 1F1B, let's consider a typical **multi-layer perceptron (MLP)** and how it is executed in a pipeline parallel setting:

<p>
$$
    \begin{align*}
        \pmb{z} &= \mathbf{W}\pmb{x}\\
        \pmb{y} &= \sigma(\pmb{z})\\
    \end{align*}
$$
</p>

where $$\pmb{x}$$ is the input, $\mathbf{W}$ is the weight matrix, $\sigma(\cdot)$ is the activation function, and $$\pmb{y}$$ is the final output.
We use the following notation:

- **Forward operation**: $F_{i,k}$ for stage $i$ and micro-batch $k$.
- **Backward operation**:
  - **$B_{i,k}$**: Gradients with respect to **inputs** for stage $i$ and micro-batch $k$.
  - **$W_{i,k}$**: Gradients with respect to **weights** for stage $i$ and micro-batch $k$.

The computation graph for this MLP is shown below:

{% include figure.liquid loading="eager" path="assets/img/2024-11-26-zero-bubble/computation_graph.svg" class="img-fluid rounded z-depth-1 p-2" caption="Computation graph for a typical MLP, including a forward pass and back-propagation." zoomable=true %}

We can see that the backward operation (both $B_{i,k}$ and $W_{i,k}$) must wait for the corresponding forward operation $F_{i,k}$ to complete.

The 1F1B strategy combines the $B_{i,k}$ and $W_{i,k}$ into a single operation, resulting in the following dependency graph:

<div class="row mt-3 align-items-end">
    <div class="col-sm-9 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-11-26-zero-bubble/dependency_graph_1.svg" class="img-fluid rounded z-depth-1" caption="Dependency graph in 1F1B." zoomable=true %}
    </div>
    <div class="col-sm-3 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-11-26-zero-bubble/pattern.svg" class="img-fluid rounded z-depth-1" caption="Wavefront pattern for pipeline scheduling." zoomable=true %}
    </div>
</div>

The dependency graph reveals a waveform pattern with three key components in pipeline scheduling:

1. **From top-left to bottom**: Process the forward pass for the same micro-batch across all stages.
2. **To the right**: Execute the backward pass at the current stage.
3. **Back up to top-right**: Propagate gradients backward to the preceding stage.

1F1B scheduling follows this pattern and consists of three phases:

1. **Warm-up phase**: Pipeline stages incrementally begin processing forward passes for the incoming micro-batches. **This phase inherently creates bubbles** (idle periods) in the pipeline as downstream stages wait for data from upstream stages.
2. **Steady state phase**: Once the pipeline is fully loaded, each stage alternates between a **forward pass** ($F$) for a new micro-batch and a **backward pass** ($B$) for a previously processed micro-batch. This steady-state execution ensures that all pipeline stages remain busy.
3. **Flush phase**: The pipeline clears all in-flight micro-batches by completing the remaining backward passes while no new forward passes are introduced. **Bubbles reappear at this stage**, as upstream stages remain idle while waiting for downstream stages to finish processing.

The figure below illustrates a 1F1B schedule with four stages (devices) and eight micro-batches. Note the wavefront pattern and the presence of bubbles:

<div class="l-body-outset">
    {% include figure.liquid loading="eager" path="assets/img/2024-11-26-zero-bubble/naive_pipeline.svg" class="img-fluid rounded z-depth-1 pt-2 pr-2" caption="1F1B pipeline scheduling." zoomable=true %}
</div>

1F1B is popular because it achieves a good balance between memory usage and throughput:

- By alternating forward and backward passes, it maintains a steady flow of computation, avoiding prolonged idle times during the steady state.
- The memory footprint is kept relatively low because the number of active micro-batches per stage is minimized.

Despite its strengths, 1F1B has inherent limitations:

1. **Tail-end bubbles in the flush phase**: During the flush phase, no new forward passes are introduced, leaving upstream stages idle as they finish their computations.
2. **Limited flexibility in operation placement**: 1F1B strictly alternates forward and backward passes, leaving no flexibility to reorder operations strategically. This rigidity means that 1F1B cannot take advantage of opportunities to fill bubbles with other computations.

In the next section, we will explore how splitting the backward pass into finer-grained components enables new schedules that significantly reduce or eliminate pipeline bubbles.

## Toward Zero Bubbles

The 1F1B schedule reduces bubbles to some extent, but it still leaves inefficiencies in the warm-up and flush phases.
Remember that the backward pass consists of two components:

1. **$B$**: Gradients with respect to **inputs**.
2. **$W$**: Gradients with respect to **weights**.

However, in 1F1B, $B$ and $W$ are grouped into a single operation, leading to sequential dependencies between $B_{i-1,k}$ and $W_{i,k}$.

The key idea behind zero-bubble pipeline is to use **finer-grained scheduling**: splitting the backward pass into $B$ and $W$, which can be scheduled independently.
This results in a refined dependency graph with more flexibility in operation placement:

<div class="row mt-3 align-items-end">
    <div class="mx-auto col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-11-26-zero-bubble/dependency_graph_2.svg" class="img-fluid rounded z-depth-1" caption="Dependency graph with finer granularity." zoomable=true %}
    </div>
</div>

Unlike $F$ and $B$, which must remain sequentially dependent, $W$ can be flexibly scheduled to fill pipeline bubbles as long as it follows its corresponding $F$.

Now, starting from the 1F1B schedule, let's split the backward passes into $B$ and $W$.
For simplicity, we assume that the execution times for $F$, $B$, and $W$ are identical.

<div class="l-body-outset">
    {% include figure.liquid loading="eager" path="assets/img/2024-11-26-zero-bubble/pipeline_1.svg" class="img-fluid rounded z-depth-1 p-2" zoomable=true %}
</div>

Thanks to the finer granularity, we can shift all operations on every stage except the last one to the left by one time step.

<div class="l-body-outset">
    {% include figure.liquid loading="eager" path="assets/img/2024-11-26-zero-bubble/pipeline_2.svg" class="img-fluid rounded z-depth-1 p-2" zoomable=true %}
</div>

We will see how the authors build on the above schedule by strategically placing operations, leading to two handcrafted schedules: **ZB-H1** and **ZB-H2**, which reduce or eliminate bubbles.

### Handcrafted Schedule 1: ZB-H1

ZB-H1 adjusts the starting points of **$W$** passes, filling the tail-end bubbles with $W$ passes without exceeding the memory usage of 1F1B.
As a result, the bubble size is reduced to approximately one-third of that in 1F1B.

To better understand ZB-H1 in action, we will derive it step by step.
First, let's remove all $W$ operations from our starting point:

<div class="mx-auto" style="max-width: 350px;">
    {% include figure.liquid loading="eager" path="assets/img/2024-11-26-zero-bubble/legend_finer.svg" class="img-fluid" zoomable=true %}
</div>

<div class="l-body-outset">
    {% include figure.liquid loading="eager" path="assets/img/2024-11-26-zero-bubble/pipeline_3.svg" class="img-fluid rounded z-depth-1 p-2" zoomable=true %}
</div>

In order to eliminate bubbles in the flush phase, we shift the last three stages to the left by one time step:

<div class="l-body-outset">
    {% include figure.liquid loading="eager" path="assets/img/2024-11-26-zero-bubble/pipeline_3_1_1.svg" class="img-fluid rounded z-depth-1 p-2" zoomable=true %}
</div>

To ensure the design of ZB-H1 meets its goals, we must carefully balance memory usage and computational dependencies while satisfying the following principles:

- The memory footprint must remain unchanged compared to the baseline.
- Sequential dependencies between $F$ and $B$ operations must be preserved.
- At least one $F$ operation must occur at every time step during the steady state phase, as backward passes ($B$) consume more memory than forward passes ($F$).
- Ideally, $W$ operations for the same micro-batch should follow a diagonal pattern, moving from the top left to the bottom right of the schedule. This spreads the workload evenly across time steps.
- There must be enough slots at each stage (eight per stage in this example) to accommodate all $W$ operations.

We'll further shift all operations to the left by reasonable amounts:

<div class="l-body-outset">
    {% include figure.liquid loading="eager" path="assets/img/2024-11-26-zero-bubble/pipeline_3_1_2.svg" class="img-fluid rounded z-depth-1 p-2" zoomable=true %}
</div>

We are almost there!
Now, let's reintroduce the $W$ operations and label their micro-batches correctly.
We also extend the schedule to show the start of the next steady state phase:

<div class="l-page-outset">
    {% include figure.liquid loading="eager" path="assets/img/2024-11-26-zero-bubble/pipeline_3_1_3.svg" class="img-fluid rounded z-depth-1 p-2" zoomable=true %}
</div>

#### Key Trade-Off

ZB-H1 significantly reduces bubbles without increasing memory usage beyond 1F1B.
However, it does not entirely eliminate bubbles, leaving some inefficiencies in the warm-up phase.

### Handcrafted Schedule 2: ZB-H2

If we are willing to slightly relax the memory constraint, we can achieve **zero bubbles** by adding extra $F$ passes during the warm-up phase and reordering $W$ passes in the flush phase.
This adjustment results in ZB-H2, which adopts a ''parallelogram-shaped'' layout that completely eliminates idle periods.

Similarly to before, we begin by removing all $W$ operations to allow flexibility in rearranging other operations:

<div class="mx-auto" style="max-width: 350px;">
    {% include figure.liquid loading="eager" path="assets/img/2024-11-26-zero-bubble/legend_finer.svg" class="img-fluid" zoomable=true %}
</div>

<div class="l-body-outset">
    {% include figure.liquid loading="eager" path="assets/img/2024-11-26-zero-bubble/pipeline_3.svg" class="img-fluid rounded z-depth-1 p-2" zoomable=true %}
</div>

We fill the warm-up phase with forward operations shifted from later time steps and move the remaining forward operations to the left accordingly:

<div class="l-body-outset">
    {% include figure.liquid loading="eager" path="assets/img/2024-11-26-zero-bubble/pipeline_3_2_1.svg" class="img-fluid rounded z-depth-1 p-2" zoomable=true %}
</div>

Now comes the crucial part: we carefully shift the steady-state operations to the left while respecting the sequential dependencies between $F$ and $B$ operations.
Moreover, we sporadically reserve spots at earlier time steps for $W$ operations to ensure a balanced memory usage.

<div class="l-body-outset">
    {% include figure.liquid loading="eager" path="assets/img/2024-11-26-zero-bubble/pipeline_3_2_2.svg" class="img-fluid rounded z-depth-1 p-2" zoomable=true %}
</div>

Finally, let's put $W$ operations back in place.
Note how the pipeline achieves zero bubbles by transitioning from a trapezoidal layout (1F1B and ZB-H1) to a parallelogram layout:

<div class="l-page-outset">
    {% include figure.liquid loading="eager" path="assets/img/2024-11-26-zero-bubble/pipeline_3_2_3.svg" class="img-fluid rounded z-depth-1 p-2" zoomable=true %}
</div>

#### Key Trade-Off

ZB-H2 eliminates pipeline bubbles entirely and maximizes throughput.
However, it requires additional memory due to:

- More active micro-batches during the warm-up phase.
- Delayed $W$ passes increasing peak memory usage.

## Automatic Zero-Bubble Scheduling

While the handcrafted schedules ZB-H1 and ZB-H2 demonstrate the power of finer-grained scheduling, they rely on idealized assumptions, such as identical execution times for $F$, $B$, and $W$.
Real-world models, however, often face variable execution times, heterogeneous hardware, and different memory constraints.
To address this, the authors propose an **automatic zero-bubble scheduling algorithm**, which generalizes into a fully automated framework.

### From Heuristics to Automation

Denote the running time of a backward pass as $T_B$.
Drawing from the derivation of ZB-H2, the automatic scheduler dynamically adjusts the placement of operations based on the following principles:

1. During the warm-up phase, schedule as many $F$ passes as possible to minimize the bubble before the first $B$.
2. After the warm-up phase, alternate between one $F$ and one $B$, inserting $W$ operations to fill bubbles when gaps larger than $T_W$ occur. Additionally, insert $W$ operations to recycle memory when the memory limit is reached. Over time, this heuristic approach stabilizes into a steady-state 1$F$-1$B$-1$W$ pattern.
3. At each stage, after all $F$ and $B$ passes are completed, schedule the remaining $W$ operations to clear the pipeline.

Moreover, the communication time required to transfer activations or gradients between stages was ignored in the previous analysis.
The authors address optimizer synchronization while maintaining synchronous semantics through an **optimistic** approach.
This assumes that most training iterations proceed without numerical issues and that most synchronization steps on global states have no significant effect.
Instead of relying on ad-hoc synchronization, the authors propose a **post-hoc update validation**:

1. Before the optimizer step at each stage, a partially reduced global state from the previous stage is combined with the current stage's local state and passed to the next stage.
2. During the warm-up phase of the next iteration, the fully reduced global state is propagated back from the last stage to the first stage.
3. Upon receiving the fully reduced global state, each stage validates whether the previous optimizer step needs to be invalidated and redone based on this complete state.

## Experiments and Results

The authors conducted comprehensive experiments to evaluate the performance of their zero-bubble scheduling strategies, comparing the handcrafted and automatically generated schedules against baseline methods.
The results demonstrate that zero-bubble scheduling consistently outperforms traditional approaches in both throughput and efficiency, with trade-offs between memory usage and performance.

The experiments were implemented using the open-source Megatron-LM framework, trained models analogous to GPT-3, and were run on up to **32 NVIDIA A100 GPUs** distributed across 4 nodes.

The authors compared the following scheduling strategies:

- **1F1B and 1F1B-I**: Standard and interleaved 1F1B schedules. The interleaved variant (1F1B-I) splits the model into smaller chunks to reduce bubbles.
- **ZB-1p**: Automatically searched schedule designed to match the peak memory usage of 1F1B while achieving much fewer bubbles.
- **ZB-2p**: Automatically searched schedule with increased memory allowance (roughly twice that of 1F1B), enabling it to achieve nearly zero bubbles.

Metrics evaluated include:

- **Throughput**: Measured as the number of samples processed per second.
- **Bubble rate**: Quantifies the proportion of idle time in the pipeline. A lower bubble rate indicates higher pipeline utilization.
- **Memory usage**: Evaluated as the peak memory consumed per GPU during training.

### Key Findings

#### Throughput Performance

**ZB-2p** consistently outperformed all other methods across various configurations, achieving throughput improvements of up to **30%** compared to 1F1B, even when using fewer micro-batches.

**ZB-1p** performed comparably to 1F1B-I in single-node setups but outperformed it in multi-node setups where communication bandwidth was a bottleneck.
Its ability to reduce pipeline bubbles without communication overhead was a key advantage.

#### Bubble Rate Analysis

**ZB-2p** achieved a bubble rate of **less than 1%** in most setups.
ZB-2p's bubble rate was consistently lower than ZB-H2, showing the effectiveness of the automatic scheduling algorithm.

**ZB-1p**'s bubble rate was comparable to ZB-H1, where memory constraints become the dominant factor in limiting improvement.

#### Memory vs. Performance Trade-Offs

**ZB-2p** achieved the best throughput but required roughly **twice** the memory of 1F1B.
Therefore, ZB-2p is more ideal for memory-rich setups.

**ZB-1p** matched the memory usage of 1F1B while achieving significant throughput gains, making it a more practical option with limited memory.

## Limitations

Although zero-bubble scheduling shows promising results, we have identified several limitations.

The handcrafted schedules ZB-H1 and ZB-H2 assume that the forward pass ($F$), backward pass for inputs ($B$), and backward pass for weights ($W$) have **identical** execution times.
In practice, these times can vary significantly across layers and stages and can introduce additional bubbles.

The automatic scheduling algorithm can struggle with highly **heterogeneous device latencies or bandwidths**.
For example, devices with slower interconnects (e.g., PCIe instead of NVLink) or highly distributed setups (e.g., across multiple servers) can cause bottlenecks.

The zero-bubble scheduling strategies assume a **synchronous** training setup, where all pipeline stages must remain in sync.
This design ensures exact optimization semantics but limits applicability in asynchronous environments.

## Conclusion

Pipeline bubbles have long been a limiting factor in distributed training, reducing throughput and leaving computational resources under-utilized.
The zero-bubble pipeline scheduling strategies presented in this paper mark a significant step forward, achieving higher throughput while maintaining synchronous training semantics.

### Contributions

We summarize the key contributions of zero-bubble scheduling:

- **Finer-grained scheduling**: By splitting the backward pass into gradients with respect to **inputs** ($B$) and **weights** ($W$), finer-grained scheduling introduces flexibility that reduces and even eliminates pipeline bubbles.
- **Automatic scheduling**: The automatic zero-bubble scheduler adapts to real-world conditions, including heterogeneous execution times, hardware configurations, and memory constraints.
- **Integration with existing strategies**: The proposed methods are orthogonal to other parallelism strategies like data parallelism (DP), tensor parallelism (TP), and ZeRO <d-cite key="rajbhandari2020zero" />, and they can be integrated into hybrid training setups for further performance gains.

### Future Work

While zero-bubble scheduling has demonstrated significant potential, we have identified several avenues for future research.

Adapting the zero-bubble approach to **asynchronous** training settings could further improve scalability by eliminating synchronization requirements.
This would require addressing challenges in managing dependencies and ensuring consistent optimization semantics.
In addition, future work could focus on extending the automatic scheduler to handle highly **heterogeneous** environments, such as clusters with varying device speeds, memory capacities, and interconnect bandwidths.

Investigating **dynamic scheduling techniques** that adapt in real-time to changing workloads or hardware conditions can optimize training efficiency.
As **hybrid parallelism strategies** become common, integrating zero-bubble scheduling into DP or TP could lead to greater performance benefits.

---

Zero-bubble pipeline scheduling represents a significant advance in distributed training, demonstrating how finer-grained scheduling can boost throughput and resource utilization.
This blog post builds on these ideas by providing detailed visualizations, contextual insights, and step-by-step clarity, making this complex topic more accessible.
We hope these contributions help others better understand and apply zero-bubble scheduling, sparking more innovation in scalable deep learning systems.
