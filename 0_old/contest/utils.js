function initNodePosition(graph){
  graph.nodes.forEach((n)=>{
    n.vx = 0;
    n.vy = 0;
    n.x = Math.random();
    n.y = Math.random();
  });
}