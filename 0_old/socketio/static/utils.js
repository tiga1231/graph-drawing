let utils = {};

utils.graph2data = (graph)=>{
  let nodes = graph.nodes.map(i=>({id:i}));
  let edges = graph.edges.map((e)=>({source:e[0], target:e[1]}));
  return {nodes, edges};
};