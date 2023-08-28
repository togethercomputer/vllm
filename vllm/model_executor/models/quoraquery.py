import requests
endpoint = 'https://api.together.xyz/inference'

bad_prompt = "<s>[INST] dfs in c++ [/INST] DFS (Depth-First Search) is a graph traversal algorithm that visits a node and then visits all of its neighbors before backtracking. Here is an example of how to implement DFS in C++:\n```\n#include <iostream>\n#include <vector>\n\nusing namespace std;\n\n// A function to perform DFS on a graph\nvoid dfs(vector<vector<int>>& graph, int start) {\n  // Create a stack to keep track of the nodes to visit\n  stack<int> toVisit;\n\n  // Add the starting node to the stack\n  toVisit.push(start);\n\n  // While the stack is not empty\n  while (!toVisit.empty()) {\n    // Get the next node to visit\n    int node = toVisit.top();\n    toVisit.pop();\n\n    // If the node has not been visited before\n    if (!graph[node].visited) {\n      // Mark the node as visited\n      graph[node].visited = true;\n\n      // Add the node's neighbors to the stack\n      for (int neighbor : graph[node]) {\n        toVisit.push(neighbor);\n      }\n    }\n  }\n}\n\nint main() {\n  // Create a graph with 5 nodes\n  vector<vector<int>> graph = {\n    {1, 2},\n    {2, 3},\n    {3, 4},\n    {4, 5},\n    {5, 1}\n  };\n\n  // Perform DFS on the graph starting at node 1\n  dfs(graph, 1);\n\n  return 0;\n}\n```\nThis code will perform DFS on the graph starting at node 1 and will visit all the nodes in the graph. The `visited` property of each node is used to keep track of which nodes have been visited. The `toVisit` stack is used to keep track of the nodes to visit. The `dfs` function takes a graph and a starting node as input and performs DFS on the graph starting at the starting node. </s><s>[INST] rewrite in python [/INST] "


res = requests.post(endpoint, json={
    "model": "togethercomputer/CodeLlama-34b-Instruct",
    "max_tokens": 512,
    "prompt": bad_prompt,
    "request_type": "language-model-inference",
    "temperature": 0.0,
    "top_p": 0.7,
    "top_k": 50,
    "repetition_penalty": 1,
    "stop": [
        "</s>",
        "[INST]"
    ],
    "safety_model": "togethercomputer/GPT-JT-Moderation-6B",
    "repetitive_penalty": 1,
    "update_at": "2023-08-27T22:31:46.589Z"
}, headers={
    "Authorization": "Bearer 803fb0bc9afcb7f048505ce0c89f19642f50639d3a97a002feaf4bec27151eaa",
})

print(res.json())