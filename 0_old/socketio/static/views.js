function initGraph(svg, data){

  let width = svg.node().clientWidth;
  let height = svg.node().clientHeight;
  let margin = 10;

  console.log('initGraph');
  if ('graph' in data){
    svg.graph = data.graph;
    svg.graph.edges.forEach((e,i)=>{
      e.source = svg.graph.nodes[e.source];
      e.target = svg.graph.nodes[e.target];
    });
  }

  if ('pos' in data){
    let xExtent = d3.extent(data.pos, d=>d[0]);
    let yExtent = d3.extent(data.pos, d=>d[1]);
    if (svg.sx === undefined){
      svg.sx = d3.scaleLinear();
      svg.sy = d3.scaleLinear()
    }else{
      xExtent[0] = Math.min(xExtent[0], svg.sx.domain()[0]);
      xExtent[1] = Math.max(xExtent[1], svg.sx.domain()[1]);
      yExtent[0] = Math.min(yExtent[0], svg.sy.domain()[0]);
      yExtent[1] = Math.max(yExtent[1], svg.sy.domain()[1]);
    }
    svg.sx
      .domain(xExtent)
      .range([margin, width-margin]);
    svg.sy
      .domain(yExtent)
      .range([height-margin, margin]);

    svg.graph.nodes.forEach((d,i)=>{
      d.x = data.pos[i][0];
      d.y = data.pos[i][1];
    });
    

    svg.nodes = svg.selectAll('.node')
    .data(svg.graph.nodes)
    .enter()
    .append('circle')
    .attr('class', 'node')
    .attr('r', 5)
    .attr('fill', 'blue'); ;
    svg.nodes = svg.selectAll('.node');

    let DURATIION = 30;
    svg.nodes
    .transition()
    .duration(DURATIION)
    .attr('cx', d=>svg.sx(d.x))
    .attr('cy', d=>svg.sy(d.y));

    svg.edges = svg.selectAll('.edge')
    .data(svg.graph.edges)
    .enter()
    .append('line')
    .attr('class', 'edge')
    .attr('stroke', '#333');
    svg.edges = svg.selectAll('.edge');

    svg.edges
    .transition()
    .duration(DURATIION)
    .attr('x1', d=>svg.sx(d.source.x))
    .attr('x2', d=>svg.sx(d.target.x))
    .attr('y1', d=>svg.sy(d.source.y))
    .attr('y2', d=>svg.sy(d.target.y));


  }


  
  



}
