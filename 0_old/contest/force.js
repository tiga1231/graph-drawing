function _forceTemplate(){
  let force = (alpha)=>{
    
  };
  force.initialize = (nodes)=>{
    this.nodes = nodes;
    return force;
  };
  force.strength = (s)=>{
    this.strength = s;
    return force;
  };
  return force;
}

//====================================


function gridForce(resolution=1){
  this.strength = 10.0;

  let force = (alpha)=>{
    for(let n of _.shuffle(this.nodes)){
      let [xTarget, yTarget] = [Math.round(n.x), Math.round(n.y)];
      n.vx = this.strength * (n.x-xTarget) * Math.pow(1-alpha, 2) * Math.pow(alpha, 4);
      n.vy = this.strength * (n.y-yTarget) * Math.pow(1-alpha, 2) * Math.pow(alpha, 4);
    }
  }
  force.initialize = (nodes)=>{
    this.nodes = nodes;
  };

  force.strength = (s)=>{
    this.strength = s;
    return force;
  };
  return force;
}


function upwardForce(edges, targetDy=1){
  this.edges = edges;
  this.strength = 1.0;

  let force = (alpha)=>{
    for(let n of this.nodes){
      n.vx *= 1-alpha;
      n.vy *= 1-alpha;
    }
    let sampleSize = this.edges.length > 50 ? 0.7*this.edges.length : this.edges.length;
    for(let e of _.sample(this.edges, sampleSize)){
      if(e.target.y - e.source.y < targetDy){
        e.target.vy += alpha * this.strength * (e.source.y - e.target.y + targetDy);
        e.source.vy -= alpha * this.strength * (e.source.y - e.target.y + targetDy);
      }
    }
  }

  force.initialize = (nodes)=>{
    this.nodes = nodes;
  };

  force.strength = (s)=>{
    this.strength = s;
    return force;
  };
  return force;
}


function stressForce(edges){
  this.edges = edges;
  this.strength = 1;
  this.weight = (a,b)=>1;
  this.targetDist = (a,b)=>1;

  let force = (alpha)=>{
    for(let n of this.nodes){
      n.vx *= 1-alpha;
      n.vy *= 1-alpha;
    }

    let sampleSize = this.edges.length > 50 ? 0.6*this.edges.length : this.edges.length;
    for(let e of _.sample(this.edges, sampleSize)){
      let w = this.weight(e.source, e.target);
      let targetDist = this.targetDist(e.source, e.target);

      let p0 = [e.source.x, e.source.y];
      let p1 = [e.target.x, e.target.y];
      let currentDist = numeric.norm2(numeric.sub(p1, p0));

      let dir = numeric.div(numeric.sub(p1, p0), currentDist + 0.1);
      let coef = (currentDist - targetDist) / 2 * w * this.strength;
      coef = math.min(coef, currentDist/2 * 0.8);
      let [dx, dy] = numeric.mul(coef, dir);
      e.source.vx += dx * alpha;
      e.source.vy += dy * alpha;
      e.target.vx += -dx * alpha;
      e.target.vy += -dy * alpha;
    }
  }

  force.initialize = (nodes)=>{
    this.nodes = nodes;
    return force;
  };

  force.weight = (accessor)=>{
    this.weight = accessor;
    return force;
  };

  force.targetDist = (accessor)=>{
    this.targetDist = accessor;
    return force;
  };

  force.strength = (s)=>{
    this.strength = s;
    return force;
  };

  return force;
}



function boundaryForce([x0,x1], [y0,y1]){
  let force = (alpha)=>{
    for(let n of this.nodes){
      if(n.x > x1){
        n.vx -= (n.x - x1);
      }else if(n.x < x0){
        n.vx += (x0 - n.x);
      }

      if(n.y > y1){
        n.vy -= (n.y - y1);
      }else if(n.y < y0){
        n.vy += (y0-n.y);
      }
    }
  };
  force.initialize = (nodes)=>{
    this.nodes = nodes;
    return force;
  };
  return force;
}

//force directed layout
function startSimulation(graph, svg){
  if(graph.graphDistance === undefined){
    graph.graphDistance = graphDistance(grpah);
  }
  
  const niter = 300;
  const defaultBoundary = [[0,10,], [0,10]];

  let simulation = d3.forceSimulation(graph.nodes)
  .alphaDecay(1 - Math.pow(0.001, 1 / niter))
  .force('link', 
    d3.forceLink(graph.edges)
    .id(d => d.id)
    .strength(0.0001)
    .distance(d=>{
      return 0.1;
    })
  )
  .force('stress', 
    stressForce(graph.edges)
    .weight((a,b)=>graph.weight[a.index][b.index])
    .targetDist((a,b)=>graph.graphDistance[a.index][b.index])
    .strength(0.001)
  )
  .force('upward', 
    upwardForce(graph.edges, 1)
    .strength(0.5)
  )
  .force('boundary', 
    boundaryForce(
      [0, graph.width || defaultBoundary[0][1]], 
      [0, graph.height || defaultBoundary[1][1]])
  )
  // .force('grid', 
  //   gridForce()
  // )
  // .force('charge', 
  //   d3.forceManyBody().strength(-0.001)
  // )
  .force('collide', 
    d3.forceCollide(1).strength(0.01)
  )
  .force('center', 
    d3.forceCenter(
      (graph.width || defaultBoundary[0][1])/2,
      (graph.height || defaultBoundary[1][1])/2)
  )
  ;
  



  simulation.on('tick', () => {
    console.log('tick');
    drawGraph(graph, svg);
  });

  return simulation;
}

