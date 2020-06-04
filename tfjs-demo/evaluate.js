function updateAxes(svg, sx, sy){
  let ax = d3.axisBottom(sx)
  .ticks(5)
  .tickSizeInner(-(sy.range()[0]- sy.range()[1]));
  let ay = d3.axisLeft(sy)
  .ticks(4)
  .tickSizeInner(-(sx.range()[1]- sx.range()[0]));
  let gx = svg.selectAll('.xAxis')
  .data([0,])
  .enter()
  .append('g')
  .attr('class', 'xAxis');
  gx = svg.selectAll('.xAxis')
  .attr('transform', `translate(${0},${sy.range()[0]})`)
  .call(ax);
  let gy = svg.selectAll('.yAxis')
  .data([0,])
  .enter()
  .append('g')
  .attr('class', 'yAxis');
  gy = svg.selectAll('.yAxis')
  .attr('transform', `translate(${sx.range()[0]},${0})`)
  .call(ay);
}

function drawGraph(graph, svg){
  if(svg.sx == undefined){
    svg.sx = d3.scaleLinear();
    svg.sy = d3.scaleLinear();
  }

  function updateScales(){
    let width = svg.node().clientWidth;
    let height = svg.node().clientHeight;

    let xExtent = d3.extent(graph.nodes, d=>d.x);
    let yExtent = d3.extent(graph.nodes, d=>d.y);
    
    svg.xDomain = xExtent;
    svg.yDomain = yExtent;
    
    xExtent = svg.xDomain.slice(0);
    yExtent = svg.yDomain.slice(0);
    let xSize = xExtent[1] - xExtent[0];
    let ySize = yExtent[1] - yExtent[0];

    let xViewport = [30, width-20];
    let yViewport = [height-20,20];
    let drawWidth = xViewport[1] - xViewport[0];
    let drawHeight = yViewport[0] - yViewport[1];

    if (drawWidth/drawHeight > xSize/ySize){
      let adjust = (ySize / drawHeight * drawWidth) - xSize;
      xExtent[0] -= adjust/2;
      xExtent[1] += adjust/2;
    }else{
      let adjust = (xSize / drawWidth * drawHeight) - ySize;
      yExtent[0] -= adjust/2;
      yExtent[1] += adjust/2;
    }
    
    svg.sx.domain(xExtent)
    .range(xViewport);
    svg.sy.domain(yExtent)
    .range(yViewport);
  }
  

  function draw(){
    svg.selectAll('.edge')
    .data(graph.edges)
    .exit()
    .remove();
    let edges = svg.selectAll('.edge')
    .data(graph.edges)
    .enter()
    .append('line')
    .attr('class', 'edge')
    .attr('fill', 'none')
    .attr('stroke', '#333')
    .attr('stroke-width', 2)
    .attr('opacity', 0.8);
    edges = svg.selectAll('.edge')
    .attr('x1', d=>svg.sx(d.source.x))
    .attr('x2', d=>svg.sx(d.target.x))
    .attr('y1', d=>svg.sy(d.source.y))
    .attr('y2', d=>svg.sy(d.target.y));

    svg.selectAll('.node')
    .data(graph.nodes)
    .exit()
    .remove();
    let newNodes = svg.selectAll('.node')
    .data(graph.nodes)
    .enter()
    .append('g')
    .attr('class', 'node')
    .call(
      d3.drag()
      .on('drag', (d)=>{
        boundaries = undefined;
        let x = d3.event.sourceEvent.offsetX;
        let y = d3.event.sourceEvent.offsetY;
        let dx = d3.event.dx;
        let dy = d3.event.dy;
        d.x = svg.sx.invert(x);
        d.y = svg.sy.invert(y);
        let newPos = graph.nodes.map(d=>[d.x, d.y]);
        dataObj.x.assign(tf.tensor2d(newPos));
        draw();
      })

    );

    let newCircles = newNodes
    .append('circle')
    .attr('r', 6)
    .attr('fill', d3.schemeCategory10[0]);

    let newTexts = newNodes
    .append('text')
    .style('font-size', 6)
    .style('fill', '#eee')
    .style('text-anchor', 'middle')
    .style('alignment-baseline', 'middle');

    let nodes = svg.selectAll('.node')
    .attr('transform', d=>`translate(${svg.sx(d.x)},${svg.sy(d.y)})`)
    .moveToFront();
    let texts = nodes.selectAll('text')
    .text(d=>d.id);
    let circles = nodes.selectAll('.circles');
  }

  window.addEventListener('resize', ()=>{
    updateScales();
    updateAxes(svg, svg.sx, svg.sy);
    draw();
  });

  updateScales();
  updateAxes(svg, svg.sx, svg.sy);
  draw();
}//drawGraph end





function loadAndEvaluate(fn, metricNames){
  let graphName = fn.split('/');
  graphName = graphName[graphName.length-1].split('.')[0];

  d3.json(fn).then((graph)=>{

    preprocess(graph);
    // let xmean = d3.mean(graph.nodes, d=>d.x);
    // let ymean = d3.mean(graph.nodes, d=>d.y);

    // graph.nodes.forEach(d=>{
    //   d.x = (d.x - xmean);// / 80;
    //   d.y = (d.y - ymean);// / 80;
    // })

    window.x = graph.nodes.map(d=>[d.x, d.y]);

    evaluateAndShow(graph, graphName, metricNames);
  });
}


function evaluateAndShow(graph, graphName, metricNames){
  let n_neighbors = graph.graphDistance
  .map((row)=>{
    return row.reduce((a,b)=>b==1?a+1:a, 0);
  });
  let adj = graph.graphDistance.map(row=>row.map(d=>d==1 ? 1.0 : 0.0));
  adj = tf.tensor2d(adj);
  let graphDistance = tf.tensor2d(graph.graphDistance);
  let stressWeight = tf.tensor2d(graph.weight);
  let edgePairs = getEdgePairs(graph);
  let neighbors = graph2neighbors(graph);
  let edges = graph.edges.map(d=>[d.source.index, d.target.index]);
  let sampleSize = 5;
  let x = graph.nodes.map(d=>[d.x, d.y]);
  x = tf.variable(tf.tensor2d(x));

  let dataObj = {
    x, 
    coef: {},
    graphDistance, 
    adj,
    stressWeight,
    graph,
    n_neighbors,
    edgePairs,
    neighbors,
    edges,
  };

  let dummy_optimizer = tf.train.momentum(0, 0, false);
  console.log(graphName);
  let {loss, metrics} = trainOneIter(dataObj, dummy_optimizer, true);
  console.log(metrics);
  console.log('=============');

  let metricsTable = d3.select('#metrics');
  let tableRow = metricsTable.append('tr');

  //title
  let nameEntry = tableRow.append('td').text(graphName);

  //graph thumbnail
  let svg = tableRow.append('td')
  .append('svg')
  .attr('width', 200*1.618)
  .attr('height', 200);
  drawGraph(graph, svg);

  //metrics
  let metricList = metricNames.map(k=>({id:k, value:metrics[k]}));
  showMetrics(metricList, tableRow);

}

function showMetrics(metricList, row){
  row.selectAll('.metric')
  .data(metricList)
  .enter()
  .append('td')
  .attr('class', 'metric');

  let td = row.selectAll('.metric');
  td.text(d=>{
    if(d.id == 'crossing_number'){
      return `${d.value.toFixed(0)}`;
    }else{
      return `${d.value.toFixed(4)}`;
    }
  });
}


function initTableHeader(keys){
  let metricsTable = d3.select('#metrics');
  let headerRow = metricsTable.append('tr')
  .attr('class', 'tableHeader');
  headerRow.selectAll('th')
  .data(['name', 'graph', ...keys])
  .enter()
  .append('th');

  headerRow.selectAll('th')
  .text(d=>d);

}

window.onload = function(){
  let keys = [
  "stress", 
  "vertex_resolution",
  "angular_resolution", 
  "aspect_ratio", 
  "crossing_angle", 
  "crossing_number", 
  "edge_uniformity", 
  "gabriel", 
  "neighbor", 
  ];
  initTableHeader(keys);

  let fns = [
    "data/neato_sfdp_layouts_json/cycle_sfdp.json",
    "data/neato_sfdp_layouts_json/cycle_neato.json",
    "data/neato_sfdp_layouts_json/bipartite_neato.json",
    "data/neato_sfdp_layouts_json/bipartite_sfdp.json",
    "data/neato_sfdp_layouts_json/spx_teaser_neato.json",
    "data/neato_sfdp_layouts_json/cube_neato.json",
    "data/neato_sfdp_layouts_json/dodecahedron_sfdp.json",
    "data/neato_sfdp_layouts_json/cube_sfdp.json",
    "data/neato_sfdp_layouts_json/nonsymmetric_neato.json",
    "data/neato_sfdp_layouts_json/dodecahedron_neato.json",
    "data/neato_sfdp_layouts_json/tree_sfdp.json",
    "data/neato_sfdp_layouts_json/block_sfdp.json",
    "data/neato_sfdp_layouts_json/grid_sfdp.json",
    "data/neato_sfdp_layouts_json/grid_neato.json",
    "data/neato_sfdp_layouts_json/block_neato.json",
    "data/neato_sfdp_layouts_json/tree_neato.json",
    "data/neato_sfdp_layouts_json/nonsymmetric_sfdp.json",
    "data/neato_sfdp_layouts_json/complete_sfdp.json",
    "data/neato_sfdp_layouts_json/complete_neato.json",
    "data/neato_sfdp_layouts_json/spx_teaser_sfdp.json",

  ];
  for(let fn of fns){
    loadAndEvaluate(fn, keys);
  }
};//onload end











