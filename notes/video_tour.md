# Script for our video
Heavily adapted from an initial script generated using ChatGPT-4o on 17.08.2024

### [Opening Scene: Introduction]

**Narrator**: "Welcome to our presentation for Mini-Challenge 3 of the VAST Challenge 2024.  Our task is to visualize and analyze a time dependend knowledge graph."

---

### [Scene 1: Starting Point for Analysis]

**Narrator**: "In 2035 SouthSeafood Express Corp was caught in illegal fishing activities and its closure has impacted the fishing industry. Therefore we start with the network containing SouthSeafood."

**Visuals**: Show the Graph View with South Seafood Company highlighted on the left. Display the scatterplot overview of all companies on the right.

---

### [Scene 2: Software Overview]

**Narrator**: "Our software consists of three main components:

1. A graph view showing the local neighborhood around selected nodes.
2. A kernel-density plot showing the frequency of events of the current sub-network.
3. A tab-layout with multiple views to identify interesting nodes.


"

**Visuals**: Briefly highlight each component: Graph View, tab-layout, and KDE plot.

---

### [Scene 3: Graph View Explanation]

**Narrator**: "The graph view provides an interactive exploration of networks. We display only an automatically-adjusted neighborhood to be able to scale different network sizes.

We use a consistent color scheme throughout the tool where Nodes and edges are color-coded based on their types. Hexagons indicate pre-identified anomalies. Additionally time is encoded via different edge markers: solid lines for active edges, dotted lines for future edges, and dashed lines for past edges.

You can see how the connection from SouthSeafood to Stiching-Marine is replaced by new edges and shell companies, while the direct path is deleted. We date this switch in the kde plot to may 2035.

"

**Visuals**: Zoom in on the Graph View, showing the different node colors, shapes, and edge markers.


---
## [Scene 5: Tab-Layout Views]
In our tab-layout we provide multiple views to identify interesting nodes.
### Scatterplots

**[downs group, miles adkinson]**
Selecting the most active nodes within the Scatterplot we believe that they are just big companies and dont seem suspicous. 
**[das zweite von oben -> graphlayout]**
However there are several outlier companies with many shareholders but without workers. The new company takes over two existing companies including all of their shareholders and beneficial owners. While the original companies had workers, the new one doesn't. This pattern repeats for all of these outliers.
**[noch weitere anklicken]**

### Heatmap
To illustrate the influence of a company's network over time, we provide a heatmap of people with the most company acquisitions and observe that the number of acquisitions increases over time. 
**[auf ein dunkelrotes klicken]**
Since, we label all nodes in the graphlayout that occur in one of our tab-views, we discover that all these companies eventually converge to a single network. 
**[Prochazka auswählen, Neighbourhood verkleinern, focus last]**
In the last year the owners acquire the same company and get rid of two of their companies. All these changes happen one month after the SouthSeafood incident.

### Multi edge view
We also identified people that become workers after already being shareholders or owners. Each glyph in this plot represents the relative timing of these relationships. Tim Hays for example owns a company for a long time and becomes the only worker 3 months after the changes of SouthSeafood. This worker relationship might be an attempt to hide prior illegal behavior without any official workers.
**[Tim Hays anklicken (oben rechts)]**

### Works for multiple
A variation of a slope graph is used to highlight people working for multiple companies. This helps in identifying potential conflicts of interest or unusual activities. The person working for the most companies is Sanaa el-Amin who is a member of the network containing SouthSeafood.
**[neighbourhood size 6]**
Particularly she works for the Stichting-Marine Company which may indicate that the other companies she works for are also involved in illegal activities. There are multiple anomalies in this network: many cycles, some with family relations, people benefitting from multiple companies with little or no workers, people working for multiple companies and the company switch associated with SouthSeafood. Therefore this whole network seems suspicious.

Thank you for watching our presentation.

--- 
