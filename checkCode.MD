


That is a highly insightful constraint. In a real-world, black-box scenario, you often **do not have the explicit caller-callee relationship** (the "trace" data) and must **infer the dependencies** (the edges) solely from time-series performance metrics (QPS, RT, Error Rate) collected from each service (the nodes).

This step is called **Dependency Graph Inference** or **Causal Discovery**, and it is where AI/ML methods truly shine.

The core principle used is **Causality and Correlation**: If Service A calls Service B, then a sudden change in Service B's metrics (especially **RT** or **Error Rate**) will often be followed, moments later, by a corresponding change in Service A's metrics.

Here is the step-by-step approach using a simple, computationally cheap method based on **Time-Lagged Cross-Correlation**, which is a practical way to infer causation from time-series data.

-----

## 🔬 Step 1: Data Preparation (Time-Series)

First, we need the historical, time-series data for our key metrics.

| Time Stamp | Service A RT | Service B RT | Service C RT | Service A QPS | ... |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $t_1$ | 120 | 50 | 80 | 500 | ... |
| $t_2$ | 125 | 55 | 80 | 510 | ... |
| $t_3$ | 130 | 100 **(Spike)** | 81 | 515 | ... |
| $t_4$ | 150 **(Propagated)** | 105 | 82 | 520 | ... |

### Python Setup for Simulated Data

We'll simulate $N$ time steps and $M$ services.

```python
import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# --- 1. Define Services (Nodes) and Metrics ---
services = ['A_Gateway', 'B_Auth', 'C_User', 'D_Product', 'E_Inventory']
N_timesteps = 100
N_services = len(services)

# Initialize a DataFrame for all metrics
data = pd.DataFrame(index=range(N_timesteps))

# Generate base noise/traffic
np.random.seed(42)
base_qps = np.cumsum(np.random.normal(500, 10, N_timesteps))
base_rt = np.cumsum(np.random.normal(100, 5, N_timesteps))

# --- Simulate Dependencies and Propagation ---
# In a real system, you'd load these from a DB. Here, we simulate the effect.
# B calls C (B depends on C). A calls B. D calls E.

for svc in services:
    # Service C (Base Layer)
    if svc == 'C_User':
        data[f'{svc}_RT'] = base_rt + np.random.normal(0, 10, N_timesteps)
    # Service B (Calls C, so B_RT is correlated with C_RT)
    elif svc == 'B_Auth':
        # B's RT is base_rt + C_RT (time from downstream) + its own work
        data[f'{svc}_RT'] = data['C_User_RT'] * 0.5 + base_rt * 0.5 + np.random.normal(10, 5, N_timesteps)
    # Service A (Calls B)
    elif svc == 'A_Gateway':
        # A's RT is base_rt + B_RT + its own work
        data[f'{svc}_RT'] = data['B_Auth_RT'] * 0.4 + base_rt * 0.6 + np.random.normal(5, 5, N_timesteps)
    # Services D and E (Separate Path)
    else:
        # Independent, simpler RTs
        data[f'{svc}_RT'] = np.random.normal(150, 15, N_timesteps)

    # All QPS are correlated to the Gateway QPS
    data[f'{svc}_QPS'] = base_qps * (random.uniform(0.5, 1.0)) + np.random.normal(0, 50, N_timesteps)
    data[f'{svc}_Error'] = np.clip(np.random.normal(0.01, 0.05, N_timesteps), 0, 0.2)
```

## 🧠 Step 2: AI/ML Inference using Cross-Correlation

We will use **Time-Lagged Cross-Correlation** on the Response Time (RT) series.

  * **Hypothesis:** If Service $Y$ calls Service $X$ (i.e., $Y \to X$), then a spike in $X$'s RT will cause a spike in $Y$'s RT **at the same time or a little later**.
  * **Measurement:** We calculate the correlation coefficient ($r$) between $RT_Y$ and $RT_X$, shifting $RT_X$ backward in time (i.e., correlating $RT_Y(t)$ with $RT_X(t - \text{lag})$).
  * **Decision Rule:** The highest positive correlation coefficient at a **positive lag** (where **$X$ precedes $Y$**) suggests that $X$ is a downstream dependency of $Y$ (i.e., $Y$ calls $X$).

### Python Implementation for Correlation Inference

```python
def infer_dependency_correlation(data, services, lag_max=5, threshold=0.6):
    inferred_dependencies = []
    
    # Iterate over every possible pair of services (Caller Candidate, Callee Candidate)
    for caller in services:
        for callee in services:
            if caller == callee:
                continue

            caller_rt = data[f'{caller}_RT']
            callee_rt = data[f'{callee}_RT']
            
            best_corr = -1
            best_lag = 0
            
            # Check for correlation at various time lags (1 to lag_max)
            for lag in range(1, lag_max + 1):
                # Align the series: correlates Caller_RT(t) with Callee_RT(t - lag)
                # If lag=1, Callee's past value is correlated with Caller's current value.
                # This implies Callee precedes Caller.
                
                # Slicing for alignment
                caller_aligned = caller_rt.iloc[lag:]
                callee_lagged = callee_rt.iloc[:-lag]
                
                # Ensure the slices are the same length
                if len(caller_aligned) == 0:
                    continue

                # Calculate Pearson Correlation
                corr, _ = pearsonr(caller_aligned, callee_lagged)
                
                if corr > best_corr:
                    best_corr = corr
                    best_lag = lag

            # If the best correlation exceeds the threshold AND the lag is positive:
            if best_corr >= threshold and best_lag > 0:
                # Dependency found: Caller -> Callee (Caller calls Callee)
                inferred_dependencies.append((caller, callee, best_corr, best_lag))
                
    return inferred_dependencies

inferred_edges = infer_dependency_correlation(data, services, lag_max=5, threshold=0.5)

print("\n--- Inferred Dependencies (Caller -> Callee) ---")
for caller, callee, corr, lag in inferred_edges:
    print(f"-> {caller} calls {callee}: Correlation={corr:.2f} at Lag={lag} (Callee precedes Caller)")

# --- 3. Construct the Inferred Graph ---
G_inferred = nx.DiGraph()
G_inferred.add_nodes_from(services)
for caller, callee, corr, lag in inferred_edges:
    G_inferred.add_edge(caller, callee, weight=corr)
```

## 🌲 Step 3: Resulting Dynamic Dependency Graph

The final step is to visualize the inferred graph. The strength of the correlation can be used as the **weight** or thickness of the edge.

```python
# --- 4. Visualize the Inferred Graph (Adding Edge Weights for Correlation) ---
def draw_inferred_graph(G, inferred_edges):
    pos = nx.circular_layout(G) 

    # Edge labels based on correlation
    edge_labels = { (u, v): f"Corr={d['weight']:.2f}" 
                    for u, v, d in G.edges(data=True) }
    
    # Edge widths based on correlation strength
    edge_widths = [d['weight'] * 5 for u, v, d in G.edges(data=True)]

    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(G, pos, node_color='lightgreen', node_size=3000, edgecolors='black')
    
    # Draw edges with width proportional to correlation
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowstyle='->', 
                           arrowsize=20, width=edge_widths)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    # Draw edge correlation labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)

    plt.title("Inferred Service Call Graph using Cross-Correlation", fontsize=16)
    plt.axis('off')
    plt.show()

draw_inferred_graph(G_inferred, inferred_edges)
```

### Summary of the AI/ML Approach

1.  **Data:** Use time series of performance metrics, primarily **Response Time (RT)**.
2.  **Model:** **Time-Lagged Cross-Correlation**.
3.  **Inference:** A high correlation between $RT_{\text{Caller}}(t)$ and $RT_{\text{Callee}}(t - \text{lag})$ implies that the Call-ee's performance impacts the Call-er's performance after a short delay, thus inferring a dependency: $\text{Caller} \to \text{Callee}$.

This resulting graph, built dynamically from metrics, is the *true* **Service Call Graph** needed for MicroHECL's next steps.

-----

The next stage in the MicroHECL methodology is to use this inferred graph for **Anomaly Detection**, where we compare the current state (like the high RT and Error Rate on Inventory) against the historical normal state. Would you like to proceed to **Step 3: Anomaly Detection**?



That's the critical next step\! After building the **Service Call Graph (SCG)**, MicroHECL (and most other AIOps solutions) must identify which of the thousands of data points are genuinely abnormal.

This is **Step 3: Anomaly Detection** for the key metrics: Response Time (RT), QPS, and Error Rate.

-----

## 🧐 Step 3: Anomaly Detection (Formalizing the Spikes)

The goal here is to formalize the visual spikes (like the red color we used earlier) by creating a statistical boundary for what is "normal" behavior.

### 1\. Model Selection: Using the Interquartile Range (IQR)

While more complex methods like Seasonal ARIMA or Deep Learning models are used in production, the **Adjusted Interquartile Range (IQR) method** is a simple, effective, and standard statistical technique for setting dynamic thresholds on metrics, especially for latency and error rates.

We calculate the **Thresholds** for each service and each metric based on historical data.

The formula for the Upper Bound (UB) and Lower Bound (LB) for a given metric time series $M$ is:

$$\text{IQR} = Q_3 - Q_1$$
$$\text{UB} = Q_3 + k \cdot \text{IQR}$$
$$\text{LB} = Q_1 - k \cdot \text{IQR}$$

Where:

  * $Q_1$ and $Q_3$ are the first (25th percentile) and third (75th percentile) quartiles of the historical data.
  * $k$ is a statistical multiplier, typically $1.5$ for outliers, but often set higher (like $k=3$ or $k=5$) for high-stakes alerting.

### 2\. Python Implementation for Anomaly Detection

We will use the simulated data from the previous step and define a "historical window" to calculate the normal bounds.

```python
import pandas as pd
import numpy as np
import networkx as nx

# Re-using the previously generated data for simplicity
# data = pd.DataFrame(...) # Assume 'data' from previous step is loaded
# G_inferred = nx.DiGraph() # Assume G_inferred is the graph from previous step

# Define parameters for IQR-based thresholding
K_MULTIPLIER = 3.0 # A common multiplier for strict outlier detection

def detect_anomalies(G, data, metrics_to_check=['RT', 'QPS', 'Error']):
    """
    Calculates statistical bounds and detects current anomalies for each service node.
    """
    anomalous_nodes = []
    
    # Use the last 20 data points as the "current state" we are checking
    current_data = data.iloc[-20:]

    for service in G.nodes:
        is_anomalous = False
        
        # We will calculate historical stats from the preceding 80 points
        historical_data = data.iloc[:-20]
        
        for metric_suffix in metrics_to_check:
            metric_col = f'{service}_{metric_suffix}'
            
            if metric_col not in historical_data.columns:
                continue

            # Calculate historical statistics (Q1, Q3, IQR)
            Q1 = historical_data[metric_col].quantile(0.25)
            Q3 = historical_data[metric_col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Calculate Upper and Lower Bounds (Thresholds)
            UB = Q3 + K_MULTIPLIER * IQR
            LB = Q1 - K_MULTIPLIER * IQR
            
            # Determine the current value (average over the last 20 steps)
            current_value = current_data[metric_col].mean()

            # Check for anomaly (High QPS/RT/Error Rate)
            if current_value > UB:
                anomaly_type = 'High'
                is_anomalous = True
            
            # Check for anomaly (Low QPS/RT, e.g., service is dead or traffic drop)
            elif current_value < LB:
                anomaly_type = 'Low'
                is_anomalous = True

            # Store the current state and anomaly status in the graph node
            G.nodes[service][f'Current_{metric_suffix}'] = round(current_value, 2)
            G.nodes[service][f'UB_{metric_suffix}'] = round(UB, 2)
            
            if is_anomalous:
                G.nodes[service]['Anomaly'] = True
                G.nodes[service][f'Anomaly_{metric_suffix}'] = f"{anomaly_type} ({round(current_value, 2)})"
                anomalous_nodes.append(service)

    return anomalous_nodes

# Execute Anomaly Detection
anomalous_services = detect_anomalies(G_inferred, data)

print("\n--- Step 3: Detected Anomalous Services ---")
if anomalous_services:
    print(f"Total Anomalous Services: {len(anomalous_services)}")
    for svc in anomalous_services:
        print(f"Service: {svc}")
        for k, v in G_inferred.nodes[svc].items():
            if 'Anomaly_' in k:
                print(f"  - {k}: {v}")
else:
    print("No significant anomalies detected in the current state.")

# --- Visualizing the Anomalies on the Graph ---
def draw_anomalous_graph(G):
    pos = nx.circular_layout(G)
    
    node_labels = {}
    node_colors = []
    
    for node in G.nodes:
        data = G.nodes[node]
        label = f"{node}\n"
        
        color = 'lightgreen'
        
        if data.get('Anomaly'):
            color = 'red'
            for metric in ['RT', 'QPS', 'Error']:
                if f'Anomaly_{metric}' in data:
                    label += f"**{metric} Anomaly:** {data[f'Current_{metric}']}\n"
        else:
            label += f"RT: {data.get('Current_RT', 'N/A')}ms\n"
        
        node_labels[node] = label
        node_colors.append(color)

    plt.figure(figsize=(10, 10))
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3500, alpha=0.9, edgecolors='black')
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowstyle='->', arrowsize=20)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, font_weight='bold')

    plt.title("MicroHECL - Service Call Graph with Anomalies (Step 3: Detection)", fontsize=16)
    plt.axis('off')
    plt.show()

# We need to ensure the data generation simulates a spike for this to work perfectly.
# Our previous data generation favored C_User and B_Auth, let's assume they spiked.
draw_anomalous_graph(G_inferred)
```




### Explanation of the Anomaly Detection

1.  **Historical vs. Current:** We divide the time series data into a large **historical window** (for calculating $Q_1$ and $Q_3$) and a small **current window** (the last 20 time steps, whose average we check).
2.  **Threshold Calculation:** The $\text{IQR}$ is calculated for each metric's historical data for a specific service. The Upper Bound ($\text{UB}$) is set using the formula. For example, for a normal service's RT:
      * $Q_1$ (25th percentile RT) = $90\text{ms}$
      * $Q_3$ (75th percentile RT) = $110\text{ms}$
      * $\text{IQR} = 20\text{ms}$
      * $\text{UB} = 110 + 3.0 \cdot 20 = 170\text{ms}$
3.  **Flagging:** If the **current average RT** for that service is $180\text{ms}$, it is flagged as an anomaly because $180\text{ms} > 170\text{ms}$.
4.  **Graph Annotation:** The graph nodes are updated to include a boolean flag (`'Anomaly': True`) and details about which metric violated the threshold.

This step generates the list of symptomatic services. The final and most crucial step of MicroHECL is to use the **inferred dependencies** (from Step 2) and the **anomalous services** (from Step 3) to identify the single, initiating **Root Cause Service**.

Would you like to proceed to **Step 4: Anomaly Propagation Chain Analysis and Root Cause Localization**?

Excellent. This is the culmination of the MicroHECL methodology—the transition from mere detection to true **Root Cause Localization**.

The core challenge here is distinguishing between the **Root Cause Service** (the initiator of the problem) and the **Symptomatic Services** (those affected by the problem).

-----

## 🔍 Step 4: Anomaly Propagation Chain Analysis and Root Cause Localization

The MicroHECL approach leverages the fact that the failure always propagates from the downstream service (callee/dependency) to the upstream service (caller).

### 1\. The Core Logic: The Downstream Precedence Rule

For any detected anomaly:

1.  If Service **A** is anomalous, and its direct downstream dependency **B** is *also* anomalous, the anomaly likely originated at **B** and propagated to **A**.
2.  If Service **C** is anomalous, but *none* of its direct downstream dependencies are anomalous, then **C** is likely the **Root Cause**.

This rule assumes that a service's problem (high RT, errors) will only affect its direct callers, not its dependencies.

### 2\. Python Implementation for Localization

We will iterate through all anomalous services and check their downstream dependencies on the inferred graph.

```python
# Re-using the G_inferred graph with 'Anomaly' flags from Step 3

def localize_root_cause(G):
    """
    Identifies the service most likely to be the root cause based on the
    downstream precedence rule and propagation analysis.
    """
    
    anomalous_services = [n for n, data in G.nodes(data=True) if data.get('Anomaly')]
    root_cause_candidates = []
    
    # 1. Check the Downstream Dependencies for Anomalies
    print("\n--- Root Cause Analysis: Checking Downstream Dependencies ---")
    
    for service in anomalous_services:
        is_root_cause = True
        
        # G.successors(service) gives the direct downstream services (Callees)
        downstream_services = list(G.successors(service))
        
        if not downstream_services:
            # If a service has no downstream dependencies (it's a leaf node), 
            # and it's anomalous, it MUST be the root cause in its path.
            print(f"Service {service}: Is anomalous and has NO downstream dependencies. **Root Cause Candidate (Leaf)**")
            root_cause_candidates.append(service)
            continue

        print(f"Service {service}: Checking its downstream services: {downstream_services}")
        
        # Check if any downstream service is also anomalous
        for downstream_svc in downstream_services:
            if G.nodes[downstream_svc].get('Anomaly'):
                # If a downstream service is anomalous, the current service is a SYMPTOM
                is_root_cause = False
                print(f"  -> {service} is a SYMPTOM. Anomaly likely propagated from its dependency: {downstream_svc}")
                break
        
        if is_root_cause:
            print(f"Service {service}: Is anomalous, but none of its dependencies are anomalous. **ROOT CAUSE DETECTED!**")
            root_cause_candidates.append(service)

    # 2. Final Root Cause Ranking (Handling multiple candidates)
    # In complex cases, you'd use a more sophisticated ranking score (e.g., based on anomaly severity).
    # For this demonstration, we'll assume the highest degree of impact (most callers affected) is the main factor,
    # or simply pick the first one found if only one is expected.
    
    if len(root_cause_candidates) > 1:
        # Simple ranking: choose the anomalous service with the highest number of direct callers (in-degree)
        final_root_cause = max(root_cause_candidates, key=lambda svc: G.in_degree(svc))
    elif len(root_cause_candidates) == 1:
        final_root_cause = root_cause_candidates[0]
    else:
        final_root_cause = "Not Found"
        
    G.graph['Root_Cause'] = final_root_cause
    return final_root_cause

# Execute Root Cause Localization
root_cause = localize_root_cause(G_inferred)

print(f"\n=====================================")
print(f"✅ MicroHECL FINAL RESULT: The Root Cause is: **{root_cause}**")
print(f"=====================================")

# --- Visualization of the Final Result ---
def draw_final_result_graph(G):
    pos = nx.circular_layout(G)
    root_cause_node = G.graph.get('Root_Cause', None)

    node_labels = {}
    node_colors = []
    
    for node in G.nodes:
        data = G.nodes[node]
        label = f"{node}\n"
        
        color = 'lightgreen'
        
        if data.get('Anomaly'):
            color = 'yellow' # Symptom color
            for metric in ['RT', 'QPS', 'Error']:
                if f'Anomaly_{metric}' in data:
                    label += f"**{metric} Anom:** {data[f'Current_{metric}']}\n"
        
        if node == root_cause_node:
            color = 'red' # Root Cause color
            label += "**ROOT CAUSE**"
            
        node_labels[node] = label
        node_colors.append(color)

    plt.figure(figsize=(10, 10))
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=4000, alpha=0.9, edgecolors='black')
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowstyle='->', arrowsize=20)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, font_weight='bold')

    plt.title("MicroHECL - Final Root Cause Localization", fontsize=16)
    plt.axis('off')
    plt.show()

draw_final_result_graph(G_inferred)
```

### Summary of Localization

1.  **Iterate Symptoms:** The algorithm only considers the services previously flagged as having an anomaly.
2.  **Check Successors (Dependencies/Callees):** For each symptomatic service (the *caller*), the algorithm checks its downstream services (the *callees*) defined by the inferred graph.
3.  **The Root Cause Test:**
      * **Symptom:** If the current service **A** is anomalous, but its dependency **B** is *also* anomalous, then **A** is merely exhibiting a symptom caused by **B**.
      * **Root Cause:** If the current service **C** is anomalous, and *none* of the services it calls (its dependencies) are anomalous, then **C** must be the source of the problem. This is because the failure stopped propagating further downstream, meaning it originated at C.
4.  **Final Ranking:** If the simple rule yields multiple candidates (e.g., two independent failure chains occurred simultaneously), you would use a final ranking factor, like the severity of the anomaly or the number of affected upstream services, to select the single most likely culprit.

This four-step process—**Graph Construction, Dependency Inference, Anomaly Detection, and Propagation Analysis**—completes the core methodology of MicroHECL for microservice error localization.

Do you have any further questions about the visualization, the statistical methods, or the interpretation of the results?
