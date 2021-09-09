const diverging_colors = ['#d73027','#f46d43','#fdae61','#dddddd','#abd9e9','#74add1','#4575b4'];

// const diverging_colors = ['#d7191c','#fdae61','#ffffbf','#abdda4','#2b83ba'];

function linspace(min, max, n=2){
  return d3.range(n).map(i=>min + (max-min) * i/(n-1));
}

function transpose(){
  // https://stackoverflow.com/questions/6297591/how-to-invert-transpose-the-rows-and-columns-of-an-html-table/40213981
  $("table").each(function() {
      var $this = $(this);
      var newrows = [];
      $this.find("tr").each(function(){
          let i = 0;
          $(this).find("th, td").each(function(){
            i++;
            if(newrows[i] === undefined) { newrows[i] = $("<tr></tr>"); }
            newrows[i].append($(this));
          });
      });
      $this.find("tr").remove();
      $.each(newrows, function(){
          $this.append(this);
      });
  });
  return false;
};


function exportPdf(){
  $('button').remove();
  print();
};




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


function drawThumbnail(graph, svg){
  if(svg.sx == undefined){
    svg.sx = d3.scaleLinear();
    svg.sy = d3.scaleLinear();
  }


  function updateScales(){
    let width = svg.node().clientWidth;
    let height = svg.node().clientHeight;

    let margin = 5;
    let xExtent = d3.extent(graph.nodes, d=>d.x);
    let yExtent = d3.extent(graph.nodes, d=>d.y);
    
    svg.xDomain = xExtent;
    svg.yDomain = yExtent;
    
    xExtent = svg.xDomain.slice(0);
    yExtent = svg.yDomain.slice(0);
    let xSize = xExtent[1] - xExtent[0];
    let ySize = yExtent[1] - yExtent[0];


    let xViewport = [margin, width-margin];
    let yViewport = [height-margin,margin];
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
    .attr('r', 1)
    // .attr('fill', d3.schemeCategory10[0]);
    .attr('fill', '#666');

    let newTexts = newNodes
    .append('text')
    .style('font-size', 0)
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
}//drawThumbnail end





// function loadAndEvaluate(fn, metricNames){
//   let graphName = fn.split('/');
//   graphName = graphName[graphName.length-1].split('.')[0];

//   d3.json(fn).then((graph)=>{

//     preprocess(graph);
//     // let xmean = d3.mean(graph.nodes, d=>d.x);
//     // let ymean = d3.mean(graph.nodes, d=>d.y);
//     // graph.nodes.forEach(d=>{
//     //   d.x = (d.x - xmean);// / 80;
//     //   d.y = (d.y - ymean);// / 80;
//     // })

//     window.x = graph.nodes.map(d=>[d.x, d.y]);

//     evaluateAndShow(graph, graphName, metricNames);
//   });
// }


function evaluateAndShow(table, graph, graphName, groupIndex, metricNames){
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

  for(let e of graph.edges){
    let i = e.source.index;
    let j = e.target.index;
    e.graph_dist = graph.graphDistance[i][j];
    e.pdist = metrics.pdist[i][j];
  }
  let tableRow = table.append('tr');

  //title
  let nameEntry = tableRow.append('td').text(graphName);

  //graph thumbnail
  let svg = tableRow.append('td')
  .append('svg')
  .attr('width', 90)
  .attr('height', 90)
  .attr('class', `group-${groupIndex}`)
  drawThumbnail(graph, svg);

  //metrics
  let metricList = metricNames.map(k=>({id:k, value:metrics[k]}));
  showMetrics(metricList, tableRow, groupIndex);
  return [tableRow, metrics];
}



function showMetrics(metricList, row, groupIndex){
  row.selectAll('.metric')
  .data(metricList)
  .enter()
  .append('td')
  .attr('class', d=>`metric ${d.id}-group-${groupIndex}`);

  let td = row.selectAll('.metric');
  td.text(d=>{
    if(d.id == 'crossing_number'){
      return `${d.value.toFixed(0)}`;
    }else{
      return `${d.value.toFixed(2)}`;
    }
  });
}


function initTableHeader(table, keys){
  let headerRow = table.append('tr')
  .attr('class', 'tableHeader');
  headerRow.selectAll('th')
  .data(['name', 'graph', ...keys])
  .enter()
  .append('th');

  headerRow.selectAll('th')
  .text(d=>d);

}


function highlightBest(groupIndex, metricNames){

  // 'stress', min
  // 'vertex_resolution', max
  // 'angular_resolution', max
  // 'aspect_ratio', max
  // 'crossing_angle', min
  // 'crossing_number', min
  // 'edge_uniformity', min
  // 'gabriel', max
  // 'neighbor', max

  metricNames.forEach(name=>{
    let data = d3.selectAll(`.${name}-group-${groupIndex}`).data();


    let best;
    if(name == 'vertex_resolution'
      || name == 'aspect_ratio'
      || name == 'gabriel'
      || name == 'neighbor'
      || name == 'angular_resolution'
    ){
      best = d3.max(data, d=>d.value);
    }else{
      best = d3.min(data, d=>d.value);
    }
    d3.selectAll(`.${name}-group-${groupIndex}`)
    .style('font-weight', d=>{
      if(d.value.toFixed(2) === best.toFixed(2)){
        return 800;
      }else{
        return 200;
      }
    });
  });
}


window.onload = function(){
  let metricNames = [
  'stress', 
  'vertex_resolution',
  'angular_resolution', 
  'aspect_ratio', 
  'crossing_angle', 
  'crossing_number', 
  'edge_uniformity', 
  'gabriel', 
  'neighbor', 
  ];

  let fnGroups = [
    [
      'data/random_layouts/cycle_random.json',
      'data/neato_layouts/cycle_neato.json',
      'data/sfdp_layouts/cycle_sfdp.json',
      // 'data/tsne_layouts/cycle_tsne.json',

      'data/gd2_layouts/cycle_GD2.json',
      'data/gd2_layouts_neato_init/cycle_GD2_neato_init.json',
      'data/gd2_layouts_sfdp_init/cycle_GD2_sfdp_init.json',
    ],
    [
      'data/random_layouts/bipartite_random.json',
      'data/neato_layouts/bipartite_neato.json',
      'data/sfdp_layouts/bipartite_sfdp.json',
      // 'data/tsne_layouts/bipartite_tsne.json',

      'data/gd2_layouts/bipartite_GD2.json',
      'data/gd2_layouts_neato_init/bipartite_GD2_neato_init.json',
      'data/gd2_layouts_sfdp_init/bipartite_GD2_sfdp_init.json',
    ],
    [
      'data/random_layouts/spx_teaser_random.json',
      'data/neato_layouts/spx_teaser_neato.json',
      'data/sfdp_layouts/spx_teaser_sfdp.json',
      // 'data/tsne_layouts/spx_teaser_tsne.json',
      
      'data/gd2_layouts/spx_teaser_GD2.json',
      'data/gd2_layouts_neato_init/spx_teaser_GD2_neato_init.json',
      'data/gd2_layouts_sfdp_init/spx_teaser_GD2_sfdp_init.json',
    ],
    [
      'data/random_layouts/cube_random.json',
      'data/neato_layouts/cube_neato.json',
      'data/sfdp_layouts/cube_sfdp.json',
      // 'data/tsne_layouts/cube_tsne.json',

      'data/gd2_layouts/cube_GD2.json',
      'data/gd2_layouts_neato_init/cube_GD2_neato_init.json',
      'data/gd2_layouts_sfdp_init/cube_GD2_sfdp_init.json',
    ],
    [
      'data/random_layouts/dodecahedron_random.json',
      'data/neato_layouts/dodecahedron_neato.json',
      'data/sfdp_layouts/dodecahedron_sfdp.json',
      // 'data/tsne_layouts/dodecahedron_tsne.json',

      'data/gd2_layouts/dodecahedron_GD2.json',
      'data/gd2_layouts_neato_init/dodecahedron_GD2_neato_init.json',
      'data/gd2_layouts_sfdp_init/dodecahedron_GD2_sfdp_init.json',
    ],
    [
      'data/random_layouts/symmetric_random.json',
      'data/neato_layouts/symmetric_neato.json',
      'data/sfdp_layouts/symmetric_sfdp.json',
      // 'data/tsne_layouts/symmetric_tsne.json',

      'data/gd2_layouts/symmetric_GD2.json',
      'data/gd2_layouts_neato_init/symmetric_GD2_neato_init.json',
      'data/gd2_layouts_sfdp_init/symmetric_GD2_sfdp_init.json',
    ],
    [
      'data/random_layouts/tree_random.json',
      'data/neato_layouts/tree_neato.json',
      'data/sfdp_layouts/tree_sfdp.json',
      // 'data/tsne_layouts/tree_tsne.json',

      'data/gd2_layouts/tree_GD2.json',
      'data/gd2_layouts_neato_init/tree_GD2_neato_init.json',
      'data/gd2_layouts_sfdp_init/tree_GD2_sfdp_init.json',
    ],
    [
      'data/random_layouts/block_random.json',
      'data/neato_layouts/block_neato.json',
      'data/sfdp_layouts/block_sfdp.json',
      // 'data/tsne_layouts/block_tsne.json',

      'data/gd2_layouts/block_GD2.json',
      'data/gd2_layouts_neato_init/block_GD2_neato_init.json',
      'data/gd2_layouts_sfdp_init/block_GD2_sfdp_init.json',
    ],
    [
      'data/random_layouts/grid_random.json',
      'data/neato_layouts/grid_neato.json',
      'data/sfdp_layouts/grid_sfdp.json',
      // 'data/tsne_layouts/grid_tsne.json',

      'data/gd2_layouts/grid_GD2.json',
      'data/gd2_layouts_neato_init/grid_GD2_neato_init.json',
      'data/gd2_layouts_sfdp_init/grid_GD2_sfdp_init.json',
    ],
    [
      'data/random_layouts/complete_random.json',
      'data/neato_layouts/complete_neato.json',
      'data/sfdp_layouts/complete_sfdp.json',
      // 'data/tsne_layouts/complete_tsne.json',

      'data/gd2_layouts/complete_GD2.json',
      'data/gd2_layouts_neato_init/complete_GD2_neato_init.json',
      'data/gd2_layouts_sfdp_init/complete_GD2_sfdp_init.json',
    ],
  ];

  colorbar();
  fnGroups.forEach((fnGroup, groupIndex)=>{
    let promises = fnGroup.map(fn=>d3.json(fn));
    Promise.all(promises)
    .then((graphs)=>{
      let table = d3.select('body')
      .append('table');
      initTableHeader(table, metricNames);

      let sc_vmax = 0;
      zip(fnGroup, graphs)
      .forEach((fn_graph_pair)=>{
        let [fn, graph] = fn_graph_pair;
        let graphName = fn.split('/');
        graphName = graphName[graphName.length-1].split('.')[0];
        preprocess(graph);
        let [row, metrics] = evaluateAndShow(table, graph, graphName, groupIndex, metricNames);
      });
      highlightBest(groupIndex, metricNames);
      colorEdges(groupIndex);
    });
  });
};//onload end


function colorEdges(groupIndex){
  let graph_count = d3.selectAll(`.group-0`)._groups[0].length;
  let edges = d3.selectAll(`.group-${groupIndex}`).selectAll('.edge');
  let data = edges.data();
  let edge_count = data.length / graph_count;

  let vmax = d3.max(data.slice(edge_count), d=>Math.abs( d.pdist - d.graph_dist)) + 0.2;
  console.log(vmax);

  // let sc = d3.scaleLinear()
  // .domain(linspace(-vmax, vmax, diverging_colors.length))
  // .range(diverging_colors);

  let sc = d3.scaleQuantile()
  .domain([-vmax, vmax])
  .range(diverging_colors);

  edges
  .attr('stroke', d=>sc(d.pdist - d.graph_dist));

}


function colorbar(){
  let sc = d3.scaleQuantile()
  .domain([-1,1])
  .range(diverging_colors);

  let width = 200;
  let height = 50;
  let svg = d3.select('body')
  .append('svg')
  .attr('width', width)
  .attr('height', height);
  // let width = d3.select('svg').node().getBoundingClientRect().width;
  // let height = d3.select('svg').node().getBoundingClientRect().width;

  let sx = d3.scaleLinear()
  .domain([0, diverging_colors.length])
  .range([20, width-20]);

  let sy = d3.scaleLinear()
  .domain([0,1])
  .range([20, height-20]);

  let g = svg
  .append('g')
  .attr('class', 'colorbar');

  g.selectAll('.color-cell')
  .data(diverging_colors)
  .enter()
  .append('rect')
  .attr('class', 'color-cell');
  let cells = g.selectAll('.color-cell')
  .attr('x', (d,i)=>sx(i))
  .attr('y', sy(0))
  .attr('width', width/diverging_colors.length)
  .attr('height', sy(1)-sy(0))
  .attr('fill', d=>d);

  let ax = d3.axisBottom(sx)
  .tickValues(numeric.add(0.5, d3.range(diverging_colors.length)))
  .tickFormat(d=>{
    if(d==0.5){
      return 'short';
    }else if(d==diverging_colors.length/2){ //just right
      return '';
    }else if (d==diverging_colors.length-0.5){
      return 'long';
    }else{
      return '';
    }
  });

  let gx = svg
  .append('g')
  .attr('transform', `translate(${0},${sy(1)})`)
  .call(ax);


}

