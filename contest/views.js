function updateAxes(svg, sx, sy){
  let ax = d3.axisBottom(sx)
  .tickSizeInner(-(sy.range()[0]- sy.range()[1]));
  let ay = d3.axisLeft(sy)
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


function updateNodePosition(graph, xy){
  graph.nodes.forEach((d,i)=>{
    d.x = xy[i][0];
    d.y = xy[i][1];
  });
}


function updateScales(graph, svg){
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


function drawGraph(graph, svg){
  if(svg.sx == undefined){
    svg.sx = d3.scaleLinear();
    svg.sy = d3.scaleLinear();
  }


  function draw(){
    let nodeRadius = 200 / graph.nodes.length;
    nodeRadius = Math.max(nodeRadius, 4); //clamp to min
    nodeRadius = Math.min(nodeRadius, 12); //clamp to max

    let arrowheadSize = 1.5;
    let a = arrowheadSize;
    svg.selectAll('#triangle')
    .data([0])
    .enter()
    .append('svg:defs')
    .append('svg:marker')
    .attr('id', 'triangle')
    .attr('refX', arrowheadSize*2)
    .attr('refY', arrowheadSize)
    .attr('markerWidth', arrowheadSize*2)
    .attr('markerHeight', arrowheadSize*2)
    .attr('orient', 'auto')
    .append('path')
    .attr('d', `M 0 0 L ${a*2} ${a} L 0 ${a*2} L ${a/2} ${a} Z`)
    .style('fill', '#333');

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
    .attr('stroke-width', 2)
    .attr('marker-end', 'url(#triangle)')
    .attr('opacity', 0.8);

    edges = svg.selectAll('.edge')
    .style('stroke', e=>e.target.y > e.source.y ? '#333':'orange')
    .attr('x1', d=>svg.sx(d.source.x))
    .attr('y1', d=>svg.sy(d.source.y))
    .attr('x2', d=>{
      let [sx,sy] = [d.source.x, d.source.y];
      let [tx,ty] = [d.target.x, d.target.y];
      let [dx,dy] = [tx-sx, ty-sy];
      let cos = dx / Math.sqrt(dx*dx + dy*dy);
      return svg.sx(d.target.x) - nodeRadius*cos * 0.9;
    })
    .attr('y2', d=>{
      let [sx,sy] = [d.source.x, d.source.y];
      let [tx,ty] = [d.target.x, d.target.y];
      let [dx,dy] = [tx-sx, ty-sy];
      let sin = dy / Math.sqrt(dx*dx + dy*dy);
      return svg.sy(d.target.y) + nodeRadius*sin * 0.9;
    });

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
        // dataObj.x.assign(tf.tensor2d(newPos));
        draw();
      })

    );

    let newCircles = newNodes
    .append('circle')
    .attr('fill', d3.schemeCategory10[0]);

    let newTexts = newNodes
    .append('text')
    .attr('class', 'node-id-text')
    .style('font-size', 6)
    .style('fill', '#eee')
    .style('text-anchor', 'middle')
    .style('alignment-baseline', 'middle');

    let nodes = svg.selectAll('.node')
    .attr('transform', d=>`translate(${svg.sx(d.x)},${svg.sy(d.y)})`)
    .moveToFront();

    let texts = svg.selectAll('.node-id-text')
    .data(graph.nodes)
    .text(d=>d.id);
    
    let circles = nodes.selectAll('circle')
    .attr('r', nodeRadius);
  }//draw end


  updateScales(graph, svg);
  updateAxes(svg, svg.sx, svg.sy);
  draw();

  window.addEventListener('resize', ()=>{
    updateScales(graph, svg);
    updateAxes(svg, svg.sx, svg.sy);
    draw();
  });

  
}//drawGraph end


function updateSvgSize(svg_graph){
  let width =  window.innerWidth/12*8 - 50;
  let height_graph =  window.innerHeight-30;
  svg_graph
  .attr('width', '100%')
  .attr('height', height_graph);
}
