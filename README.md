# Graph Theory for Social-Media Behavior and Emotion

An exploratory graph-theory project that models relationships between social-media users, platforms, activity patterns, and reported emotions. It applies graph construction, similarity networks, homophily, centrality, community detection, bipartite modeling, and shortest-path analysis to a social-media well-being dataset.

## Project contents

- `analysis.py` contains data-cleaning helpers, descriptive statistics, visualizations, and the graph algorithms and models described below.
- `archive/train.csv`, `archive/val.csv`, and `archive/test.csv` contain the dataset splits. The current script loads `archive/test.csv` directly; it does not train a machine-learning model or use the other two splits in its active workflow.
- `Report.pdf` and `Social_Media_Well_Being.odp` are the accompanying written report and presentation.
- `myenv/` is a local Python virtual environment and is excluded from Git.

## Graph models and algorithms

The project explores several ways to represent the dataset as graphs. These models are analytical abstractions of the supplied records; an edge represents a relationship defined by the code, not necessarily a real-world social connection.

### User–platform graph

The active workflow constructs an undirected NetworkX graph with user and platform nodes. Each user is linked to the platform in their record, and the edge carries the record's dominant-emotion label. A custom layout places platforms around a large circle and arranges each platform's users around that platform, forming visible user clusters centered on their associated platforms. This graph is visualized by the default run.

### Activity-similarity network and emotion homophily

An optional function scales four behavioral features—comments received, likes received, posts, and messages sent—with `MinMaxScaler`, then computes pairwise cosine similarity. It connects users whose similarity exceeds a fixed threshold of `0.9`. On this graph, it calculates the average clustering coefficient and applies the Girvan–Newman edge-removal community algorithm to obtain a community split. It then compares the fraction of edges whose endpoints fall in the same broad emotion group (positive/neutral versus negative) with an expected-agreement baseline to calculate a homophily index. The edge grouping and baseline use different category granularities, so interpret this index cautiously; it is exploratory, not a definitive statistical test.

### Weighted user–activity graph and betweenness centrality

Another optional routine builds a heterogeneous graph of users and activity-value nodes (for example, a particular number of likes or messages). Edges are weighted by the corresponding activity value. It computes normalized betweenness centrality, selects high-centrality activity nodes, and attempts to compare their associated emotion labels. Betweenness centrality measures how often a node lies on shortest paths between other nodes, giving a way to explore which activity-value nodes are structurally important in this constructed network.

### Age-group/platform bipartite graph

The age-group routine creates a bipartite graph with age bands on one side and social platforms on the other. Edges represent the platform selected by the routine's emotion filter for each age band. The result is displayed with a bipartite layout to compare platform associations across groups.

### Activity-to-emotion paths

The shortest-path routine creates a directed, layered graph from platform to likes, messages, comments, posts, and a selected emotion. It uses NetworkX's weighted shortest-path interface to search for a route to the target emotion. The edges and weights in this constructed graph encode activity values; they should be understood as a modeling exercise, not as evidence that one activity causes an emotion.

### Descriptive and distributional analysis

The script also reports ranges, means, variances, and standard deviations for numeric fields. Optional plotting code summarizes platform frequency, age bands, and dominant emotions.

## Default execution

When run as a script, `analysis.py` loads `archive/test.csv`, cleans duplicate user records, prints descriptive statistics, and displays the user–platform graph. The other graph analyses and distribution plots are implemented as functions but are not called in the default `__main__` workflow. To explore one, load and prepare a dataframe and call its function from the main section.

## Requirements

Python 3 and the following packages are imported by `analysis.py`:

- pandas
- NumPy
- Matplotlib
- NetworkX
- scikit-learn

Install the dependencies in an environment of your choice:

```bash
python -m pip install pandas numpy matplotlib networkx scikit-learn
```

## Run

Run the script from the project directory so its relative CSV path resolves:

```bash
python analysis.py
```

The active workflow reads `archive/test.csv`, prints descriptive statistics, and opens the user–platform graph visualization. Close the plot window to finish the process. The optional graph algorithms can be enabled by calling their corresponding functions after loading and preparing the dataframe.

## Data columns

The CSV data includes a user identifier, age, gender, platform, daily usage time, daily posts, likes received, comments received, messages sent, and dominant emotion. The supplied train/validation files label the usage column `Daily_Usage_Time (minutes)`, while the test file labels it `Daily_Usage_Time`; code that switches between splits should account for this naming difference.

## Notes

- Results and graph layouts can vary depending on the data and plotting environment.
- The graph structures and metrics depend on modeling choices such as the similarity threshold, edge definitions, and weights. Interpret them in that context; they do not establish causal relationships.
- The optional routines are exploratory implementations and may need adjustment for different input data or edge cases.
- The project currently has no dependency lockfile or automated test suite.
- `myenv/` is machine-local and should not be committed; recreate the environment and install the listed packages on another machine.
