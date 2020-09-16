function updateAxes(svg, sx, sy, intGrid=false){
  let ax = d3.axisBottom(sx)
  .tickSizeInner(-(sy.range()[0]- sy.range()[1]));
  let ay = d3.axisLeft(sy)
  .tickSizeInner(-(sx.range()[1]- sx.range()[0]));
  if(intGrid){
    ax.ticks(Math.min(500, sx.domain()[1] -  sx.domain()[0]));
    ay.ticks(Math.min(500, sy.domain()[1] -  sy.domain()[0]));
  }
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


let historicalWorst = {};
let worstBound = {
  stress: Infinity,
  crossing_angle: 1,
  vertex_resolution: 0,
  edge_uniformity: Infinity,
  neighbor: 0,
  angular_resolution: 0,
  aspect_ratio: 0,
  gabriel: 0,
  crossing_number: Infinity,
  upwardness: Infinity,
};
let bestBound = {
  stress: 0,
  crossing_angle: 0,
  vertex_resolution: 1,
  edge_uniformity: 0,
  neighbor: 1,
  angular_resolution: 1,
  aspect_ratio: 1,
  gabriel: 1,
  crossing_number: 0,
  upwardness: 0,
};

function drawProperty(svg, metricHistory, name){

  let xmin = 0;
  let xmax = metricHistory.length-1;
  let ymin;
  if(worstBound[name] === -Infinity 
    || worstBound[name] === Infinity
  ){
    if(worstBound[name] === -Infinity){
      ymin = d3.min(metricHistory);
    }else if(worstBound[name] === Infinity){
      ymin = d3.max(metricHistory);
    }
    if(name in historicalWorst){
      if(worstBound[name] === -Infinity){
        historicalWorst[name] = Math.min(historicalWorst[name], ymin);
      }
      else if(worstBound[name] === Infinity){
        historicalWorst[name] = Math.max(historicalWorst[name], ymin);
      }
    }else{
      historicalWorst[name] = ymin;
    }
    ymin = historicalWorst[name];
  }else{
    ymin = worstBound[name];
  }
  let ymax = bestBound[name];

  let pathData = metricHistory.map((y,i)=>({x:i, y:y}));
  pathData.unshift({x:0, y:ymin});
  pathData.push({x:xmax, y:ymin});


  let width = svg.node().getBoundingClientRect().width;
  let height = svg.node().getBoundingClientRect().height;
  svg
  .attr('width', width)
  .attr('height', height);
  let sx = d3.scaleLinear().domain([xmin, xmax]).range([0, width]);
  let sy = d3.scaleLinear().domain([ymin, ymax]).range([height, 0]);

  let line = d3.line()
  .x((d,i)=>sx(d.x))
  .y((d,i)=>sy(d.y))
  .curve(d3.curveLinear);

  svg.selectAll('.metric-line')
  .data([pathData, ])
  .enter()
  .append('path')
  .attr('class', 'metric-line');

  let metricLine = svg.selectAll('.metric-line')
  .attr('d', d=>line(d));

  //TODO: better tool tip
  svg.selectAll('.tool-tip')
  .data([pathData, ])
  .enter()
  .append('text')
  .attr('class', 'tool-tip');

  let tooltip = svg.selectAll('.tool-tip')
  .attr('x', sx((0+xmax)/2))
  .attr('y', sy((ymin+ymax)/2))
  .text(d=>{
    return (d[d.length-2].y).toFixed(2);
  });


}
 

function traceMetrics(svgs, metricsHistory, length){
  svgs.each(function(){
    let svg = d3.select(this);
    let name = svg.attr('for');
    let dummyValue = 0.0;
    let metricHistory = metricsHistory.map(d=>{
      if(name in d){
        let value = d[name];
        dummyValue = value;
        return value;
      }else{
        return dummyValue;
      }
    });
    drawProperty(svg, metricHistory, name);
  });

}


function traceLoss(svg, losses, maxPlotIter){
  let sx = d3.scaleLinear();
  let sy = d3.scaleLinear();

  function updateScales(){
    let width = svg.node().clientWidth;
    let height = svg.node().clientHeight;
    sx.domain([0, maxPlotIter])
    .range([40, width-20]);
    sy.domain(d3.extent(losses))
    .range([height-30,20]);
  }

  

  function draw(){
    svg.selectAll('#lossCurve')
    .data([losses])
    .enter()
    .append('path')
    .attr('id', 'lossCurve');
    let lossCurve = svg.select('#lossCurve');

    lossCurve
    .attr('fill', 'none')
    .attr('stroke', d3.schemeCategory10[0])
    .attr('stroke-width', 1.5)
    .attr('d', d3.line()
      .curve(d3.curveLinear)
      .x((d,i)=>sx(i))
      .y((d)=>sy(d))
    );
  }

  window.addEventListener('resize', ()=>{
    updateScales();
    updateAxes(svg, sx, sy);
    draw();
  });
  updateScales();
  updateAxes(svg, sx, sy);
  draw();
}//traceLoss end



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
  if(svg.sx == undefined || svg.sy == undefined){
    svg.sx = d3.scaleLinear();
    svg.sy = d3.scaleLinear();
  }
  // let sx = (x)=>svg.sx(x);
  // let sy = (y)=>svg.sy(y);

  function draw(){
    let nodeRadius = 250 / graph.nodes.length;
    nodeRadius = Math.max(nodeRadius, 4); //clamp to min
    nodeRadius = Math.min(nodeRadius, 16); //clamp to max

    let arrowheadSize = 1.5;
    // let arrowheadSize = 0;
    let a = arrowheadSize;
    let r = nodeRadius;
    
    svg.selectAll('#triangle')
    .data([0])
    .enter()
    .append('svg:defs')
    .append('marker')
    .attr('id', 'triangle');
    
    svg.selectAll('#triangle')
    .attr('refX', 2*a+r/2-1)
    .attr('refY', a)
    .attr('markerWidth', a*2)
    .attr('markerHeight', a*2)
    .attr('orient', 'auto')
    .append('path')
    // .attr('d', `M 0 0 L ${2*a} ${a} L 0 ${2*a} L ${0.5*a} ${a} Z`)
    .attr('d', `M 0 0 L ${2*a} ${a} L 0 ${2*a} L ${0.5*a} ${a} Z`)
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
    // .style('stroke', '#333')
    .style('stroke-linecap', 'butt')
    .attr('x1', d=>svg.sx(d.source.x))
    .attr('y1', d=>svg.sy(d.source.y))
    .attr('x2', d=>{
      return svg.sx(d.target.x);
    })
    .attr('y2', d=>{
      return svg.sy(d.target.y);
    });

    svg.selectAll('.node')
    .data(graph.nodes)
    .exit()
    .remove();

    let isPlaying0;
    let newNodes = svg.selectAll('.node')
    .data(graph.nodes)
    .enter()
    .append('g')
    .attr('class', 'node');
    svg.selectAll('.node')
    .call(
      d3.drag()
      .on('start', ()=>{
        isPlaying0 = isPlaying;
        playButton.on('click')(false);
      })
      .on('drag', (d,i)=>{
        boundaries = undefined;
        let x = d3.event.sourceEvent.offsetX;
        let y = d3.event.sourceEvent.offsetY;
        let dx = d3.event.dx;
        let dy = d3.event.dy;
        d.x = svg.sx.invert(x);
        d.y = svg.sy.invert(y);

        dataObj.x_arr[d.index] = [d.x/graph.scalingFactor+graph.xmin, d.y/graph.scalingFactor + graph.ymin];
        dataObj.x.assign(tf.tensor2d(dataObj.x_arr));
        draw();
      })
      .on('end', (d,i)=>{
        dataObj.x_arr[d.index] = [d.x/graph.scalingFactor+graph.xmin, d.y/graph.scalingFactor+graph.ymin];
        dataObj.x.assign(tf.tensor2d(dataObj.x_arr));
        if(graph.snapToInt){
          d.x = Math.round(d.x);
          d.y = Math.round(d.y);
          draw();
        }
        playButton.on('click')(isPlaying0);
        // let newPos = graph.nodes.map(d=>[d.x, d.y]);
        // dataObj.x.assign(tf.tensor2d(newPos).div(graph.scalingFactor));
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
    .attr('transform', d=>{
      let x = svg.sx(d.x);
      let y = svg.sy(d.y);
      return `translate(${x},${y})`;
    })
    .moveToFront();

    let texts = svg.selectAll('.node-id-text')
    .data(graph.nodes)
    .text(d=>d.id);
    
    let circles = nodes.selectAll('circle')
    .attr('r', nodeRadius);
  }//draw end

  updateScales(graph, svg);
  updateAxes(svg, svg.sx, svg.sy, true);
  draw();

  window.addEventListener('resize', ()=>{
    updateScales(graph, svg);
    updateAxes(svg, svg.sx, svg.sy, true);
    draw();
  });

  
}//drawGraph end


function updateSvgSize(svg_loss, svg_graph){
  // let width =  window.innerWidth;
  let height_graph =  window.innerHeight/4*3-20;
  let height_loss =  window.innerHeight/4*1-20;
  svg_loss
  .attr('width', '100%')
  .attr('height', height_loss);
  svg_graph
  .attr('width', '100%')
  .attr('height', height_graph);
}
