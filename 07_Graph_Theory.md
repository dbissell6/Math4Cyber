# Graph Theory

Graph can be used for a couple different things. like path traversal and grouping. One of the highlights of graph it is the most visual. For some analyses you need to create a chart to demonstrate the immportant
takeaways, in 

Path traversal, moving in physical space to find the shortest path from point A to B.


Groups - graph built on linear algebra concepts allows us to look at relationships of high dimesnioanl objects. How 'far' these objects are away from eachother can help us to identify groups and patterns.

Networking. Looking at something like how resilient a network is can be done by graph. 

2 kinds of objects, nodes (The objects) and edge(the relationships to of one node to another).

## Nodes

### Attributes

## Edges

### Directional - Bidirectional


## Path Traversal example

Imagine our boss wants to know the shortest path from a workstation to another. How many steps does it take to get from workstation 19 to workstation 17.
We can use graph

![image](https://github.com/dbissell6/Math4Cyber/assets/50979196/4bae44fa-0bc8-4241-b890-7d15e347c0cf)

![image](https://github.com/dbissell6/Math4Cyber/assets/50979196/80146985-7259-4b37-b948-fa7634b2635d)


Bredth First Search can be used to show how many steps it takes from initial to anyother.

Color key

![image](https://github.com/dbissell6/Math4Cyber/assets/50979196/5df85cfd-d079-43a7-a443-d2a618138e18)

![image](https://github.com/dbissell6/Math4Cyber/assets/50979196/54c3b234-b045-49b5-a5ff-683d07782266)


More involved example from HTB CTF - `https://github.com/dbissell6/DFIR/blob/main/WalkThroughs/Cyber_Apocalypse_2024.md#path-of-survival---hard`


## Dataframes, adjaceny matrix(heatmap), graph

Lets imagine your boss wanted you to analyze some security data


![image](https://github.com/dbissell6/Math4Cyber/assets/50979196/c82854d8-c8c4-48af-82cc-1347eea3abf6)

For the past couple years every month your company kept statistics on some security variables.

```
External_Threat_Intelligence_Alerts

Internal_Security_Audits

User_Awareness_Training

Endpoint_Security_Compliance

Detected_Security_Incidents

```

(This is all pseudo data, dont try to make any real conclusions from this, demonstration purposes only)

As we learned back in stats, to compare the relationship of two variables we can use covariance, or get the normilzed version, correlation.

Heatmap

![image](https://github.com/dbissell6/Math4Cyber/assets/50979196/4cde0eef-a3f0-4d94-b977-77b4409c1496)


Looking at the heatmap it illustrates two important concepts mentioned before.

a variable correlated to itself will always be 1, that is why we see the diagnoal line.

If the information is mirrored along that diagnoal line we know the graph is not bidirectional.

### to graph

the variables are our nodes, and the correlations are the edges.

Seeing the same thing with graph

![image](https://github.com/dbissell6/Math4Cyber/assets/50979196/98dd5a7f-fef7-491d-be10-7e437a1c7415)


