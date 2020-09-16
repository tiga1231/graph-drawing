// main
let isPlaying = true;
// let dataRoot = './data/contest-prep-1';
let dataRoot = './data/contest-1';
// let dataRoot = './data';

window.onload = function(){
  let graphTypeSelect = d3.select('#graphType');
  let fn = `${dataRoot}/${graphTypeSelect.node().value}.json`;

  let snapCheckbox = d3.select('#snapCheckbox');
  let scaleSlider = d3.select('#scaleSlider');
  let scaleLabel = d3.select('#scaleLabel');

  let lrSlider = d3.select('#lr');
  window.lr0 = +Math.exp(lrSlider.node().value);
  window.lr = +Math.exp(lrSlider.node().value);
  window.lrSlider = lrSlider;
  let lrText = d3.select('#lrText')
  .text(lr.toFixed(4));

  let momentumSlider = d3.select('#momentum');
  window.momentum = +momentumSlider.node().value;
  let momentumText = d3.select('#momentumText')
  .text(momentum);

  let stressSlider = d3.select('#stressSlider');
  let stressLabel = d3.select('#stressLabel');
  let crossingAngleSlider = d3.select('#crossingAngleSlider');
  let crossingAngleLabel = d3.select('#crossingAngleLabel');
  let angularResolutionSlider = d3.select('#angularResolutionSlider');
  let angularResolutionLabel = d3.select('#angularResolutionLabel');
  let neighborSlider = d3.select('#neighborSlider');
  let neighborLabel = d3.select('#neighborLabel');
  let edgeUniformitySlider = d3.select('#edgeUniformitySlider');
  let edgeUniformityLabel = d3.select('#edgeUniformityLabel');
  let crossingNumberSlider = d3.select('#crossingNumberSlider');
  let crossingNumberLabel = d3.select('#crossingNumberLabel');
  let upwardnessSlider = d3.select('#upwardnessSlider');
  let upwardnessLabel = d3.select('#upwardnessLabel');

  let vertexResolutionSlider = d3.select('#vertexResolutionSlider');
  let vertexResolutionLabel = d3.select('#vertexResolutionLabel');

  let gabrielSlider = d3.select('#gabrielSlider');
  let gabrielLabel = d3.select('#gabrielLabel');
  let aspectRatioSlider = d3.select('#aspectRatioSlider');
  let aspectRatioLabel = d3.select('#aspectRatioLabel');

  let coef = {
    stress: +stressSlider.node().value,
    crossing_angle: +crossingAngleSlider.node().value,
    neighbor: +neighborSlider.node().value,
    edge_uniformity: +edgeUniformitySlider.node().value,
    crossing_number: +crossingNumberSlider.node().value,
    angular_resolution: +angularResolutionSlider.node().value,
    vertex_resolution: +vertexResolutionSlider.node().value,
    gabriel: +gabrielSlider.node().value,
    aspect_ratio: +aspectRatioSlider.node().value,
    upwardness: +upwardnessSlider.node().value,
  };

  let sampleSize = 5;
  let maxPlotIter = 50;
  let niter = 200000;
  let maxMetricSize = 10;

  let optimizers = [tf.train.momentum(lr, momentum, false)];
  window.optimizers = optimizers;
  let losses = [];
  let metrics = [];
  window.metrics = metrics;
  let animId = 0;

  let svg_loss = d3.select('#loss')
  .style('background-color', '#f3f3f3');

  let svg_graph = d3.select('#graph');
  window.svg_graph = svg_graph;

  let svg_metrics = d3.selectAll('.metric');

  let playButton = d3.select('#play');
  window.playButton = playButton;
  let resetButton = d3.select('#reset');
  let downloadButton = d3.select('#download');

  let initOption = d3.select('#initOption');
  let initMode = initOption.node().value;
  initOption.on('change', function(){
    initMode = d3.select(this).node().value;
  });
  updateSvgSize(svg_loss, svg_graph);
  window.addEventListener('resize', ()=>{updateSvgSize(svg_loss, svg_graph)});



  let graphJson = d3.select('#graphJson')
  .on('change', ()=>{
    let graph = JSON.parse(graphJson.node().value);
    window.graph = graph;
    cancelAnimationFrame(dataObj.animId);
    loadGraph(graph);
    graphTypeSelect.node().value = 'custom';
  });


  let graphFile = d3.select('#graphFile')
  .on('change', ()=>{
    let file = event.target.files[0];
    let reader = new FileReader();
    reader.addEventListener('load', (event) => {
      cancelAnimationFrame(dataObj.animId);

      let text = event.target.result;
      let graph = JSON.parse(text);
      window.graph = graph;
      loadGraph(graph);
      graphJson.node().value = text;
      graphTypeSelect.node().value = 'custom';
    });
    reader.readAsText(file);
  });


  
  function loadGraph(graph){
    window.graph = graph;

    let x;
    function reset(){
      console.log('reset');
      if(initMode == 'neato' && graph.initPosition_neato !== undefined){
        x = tf.variable(tf.tensor2d(graph.initPosition_neato).div(100));
      }else if(initMode == 'sfdp' && graph.initPosition_sfdp !== undefined){
        x = tf.variable(tf.tensor2d(graph.initPosition_sfdp).div(100));
      }else if(initMode == 'tsne' && graph.initPosition_tsne !== undefined){
        x = tf.variable(tf.tensor2d(graph.initPosition_tsne));
      }else if(initMode == 'random'){
        x = tf.variable(
          tf.randomUniform([graph.nodes.length,2], -1, 1)
        );
      }
      boundaries = undefined;
      historicalWorst = {};
      if(lrSlider.on('input')){
        lrSlider.on('input')(lr0);
      }
      return x;
    }
    reset();

    optimizers[0] = tf.train.momentum(lr, momentum, false);
    preprocess(graph, x.arraySync());


    let n_neighbors = graph.graphDistance
    .map((row)=>{
      return row.reduce((a,b)=>b==1?a+1:a, 0);
    });
    let adj = graph.graphDistance.map(row=>row.map(d=>d==1.0 ? 1.0 : 0.0));
    adj = tf.tensor2d(adj);
    let graphDistance = tf.tensor2d(graph.graphDistance);
    let stressWeight = tf.tensor2d(graph.weight);
    let edgePairs = getEdgePairs(graph);
    let neighbors = graph2neighbors(graph);
    let edges = graph.edges.map(d=>[d.source.index, d.target.index]);

    let dataObj = {
      sampleSize,
      x, 
      graphDistance, 
      adj,
      stressWeight,
      graph,
      animId,
      coef,
      n_neighbors,
      edgePairs,
      neighbors,
      edges,
    };
    window.dataObj = dataObj;

    function play(){
      train(dataObj, niter, optimizers, (record)=>{
        metrics.push(record.metrics);
        losses.push(record.loss);
        if (losses.length>maxPlotIter){
          losses = losses.slice(losses.length-maxPlotIter);
        }
        if (metrics.length > maxMetricSize){
          metrics = metrics.slice(metrics.length-maxMetricSize);
        }
        traceLoss(svg_loss, losses, maxPlotIter);
        traceMetrics(svg_metrics, metrics, maxMetricSize);
        if (losses.length >= 10){
          let n = losses.length;
          let firstSlice = losses.slice(Math.floor(n/2), Math.floor(n/4*3));
          let secondSlice = losses.slice(Math.floor(n/4*3), n);
          let avgLoss0 = math.mean(firstSlice);
          let avgLoss1 = math.mean(secondSlice);
          if(avgLoss1 > avgLoss0){
            lrSlider.on('input')(Math.max(lr*0.999, 0.001));
          }
        }
        niter -= 1;
        if(niter % 2 == 0){//update graph display every 2 iterations
          dataObj.x_arr = x.arraySync();
          let x_arr = postprocess(dataObj.x_arr, graph);
          updateNodePosition(graph, x_arr);
          drawGraph(graph, svg_graph);
        }
      });
    }
    play();
    // 
    // let simulation = initSimulation(graph, graph.graphDistance, svg_graph);
    

    //interactions
    
    playButton.on('click', function(shouldPlay){
      if(shouldPlay === undefined){
        isPlaying = !isPlaying;
      }else{
        isPlaying = shouldPlay;
      }
      if(isPlaying){
        playButton.attr('class', 'fas fa-pause-circle');
        play();
      }else{
        playButton.attr('class', 'fas fa-play-circle');
      }
    });

    resetButton.on('click', function(){
      let x = reset();      
      svg_graph.xDomain = undefined;
      let xy = x.arraySync();
      graph.nodes.forEach((d,i)=>{
        d.x = xy[i][0];
        d.y = xy[i][1];
      });
      dataObj.x = x;
      drawGraph(graph, svg_graph);
    });

    downloadButton.on('click', ()=>{

      // download json
      let output = {};
      output.nodes = graph.nodes.map(d=>{
        return {
          id: d.id,
          x: d.x,
          y: d.y
        }
      });
      output.edges = graph.edges.map(e=>{
        return {
          source: e.source.id,
          target: e.target.id,
        };
      });
      output.width = graph.width;
      output.height = graph.height;

      let fn_out = `${graphTypeSelect.node().value}-out.json`;
      exportJson(output, fn_out);


      // download PNG:
      // downloadPNG();
    });

    graphTypeSelect.on('change', function(){
      let fn = d3.select(this).node().value;
      fn = `${dataRoot}/${fn}.json`;
      cancelAnimationFrame(dataObj.animId);
      loadJson(fn);
      snapCheckbox.node().checked = false;
    });


    snapCheckbox.on('click', function(){
      graph.snapToInt = d3.select(this).node().checked;
      let x_arr = postprocess(dataObj.x_arr, graph);
      updateNodePosition(graph, x_arr);
      drawGraph(graph, svg_graph);
    });


    scaleSlider.on('input', function(value){
      value = value || +d3.select(this).node().value;
      graph.scalingFactor = value;
      scaleLabel.text(value.toFixed(2));
      let x_arr = postprocess(dataObj.x_arr, graph);
      updateNodePosition(graph, x_arr);
      drawGraph(graph, svg_graph);
    });

    lrSlider.on('input', function(value){
      window.lr = value || Math.exp(+d3.select(this).node().value);
      lrText.text(lr.toFixed(4));
      optimizers[0] = tf.train.momentum(lr, momentum, false);
      lrSlider.property('value', Math.log(lr));
    });

    momentumSlider.on('input', function(){
      window.momentum = +d3.select(this).node().value;
      momentumText.text(window.momentum);
      optimizers[0] = tf.train.momentum(window.lr, momentum, false);
    });

    stressSlider.on('input', function(){
      let value = +stressSlider.node().value;
      stressLabel.text(value.toFixed(2));
      coef['stress'] = value;
    });
    stressSlider.on('input')();

    crossingAngleSlider.on('input', function(){
      let value = +crossingAngleSlider.node().value;
      crossingAngleLabel.text(value.toFixed(2));
      coef['crossing_angle'] = value;
    });
    crossingAngleSlider.on('input')();

    neighborSlider.on('input', function(){
      let value = +neighborSlider.node().value;
      neighborLabel.text(value.toFixed(2));
      coef['neighbor'] = value;
    });
    neighborSlider.on('input')();

    edgeUniformitySlider.on('input', function(){
      let value = +edgeUniformitySlider.node().value;
      edgeUniformityLabel.text(value.toFixed(2));
      coef['edge_uniformity'] = value;
    });
    edgeUniformitySlider.on('input')();

    crossingNumberSlider.on('input', function(){
      let value = +crossingNumberSlider.node().value;
      crossingNumberLabel.text(value.toFixed(2));
      coef['crossing_number'] = value;
    });
    crossingNumberSlider.on('input')();

    angularResolutionSlider.on('input', function(){
      let value = +angularResolutionSlider.node().value;
      angularResolutionLabel.text(value.toFixed(2));
      coef['angular_resolution'] = value;
    });
    angularResolutionSlider.on('input')();

    vertexResolutionSlider.on('input', function(){
      let value = +vertexResolutionSlider.node().value;
      vertexResolutionLabel.text(value.toFixed(2));
      coef['vertex_resolution'] = value;
    });
    vertexResolutionSlider.on('input')();

    gabrielSlider.on('input', function(){
      let value = +gabrielSlider.node().value;
      gabrielLabel.text(value.toFixed(2));
      coef['gabriel'] = value;
    });
    gabrielSlider.on('input')();

    aspectRatioSlider.on('input', function(){
      let value = +aspectRatioSlider.node().value;
      aspectRatioLabel.text(value.toFixed(2));
      coef['aspect_ratio'] = value;
    });
    aspectRatioSlider.on('input')();

    upwardnessSlider.on('input', function(){
      let value = +upwardnessSlider.node().value;
      upwardnessLabel.text(value.toFixed(2));
      coef['upwardness'] = value;
    });
    upwardnessSlider.on('input')();
  }//loadGraph end

  function loadJson(fn){
    d3.json(fn).then((graph)=>{
      graphJson.node().value = JSON.stringify(graph, null, 2);
      loadGraph(graph);
    });
  }

  loadJson(fn);
};//onload end


// donwload svg as png:
// https://spin.atomicobject.com/2014/01/21/convert-svg-to-png/

function styles(dom) {
  var used = "";
  var sheets = document.styleSheets;
  for (var i = 0; i < sheets.length; i++) {
    try{
      var rules = sheets[i].cssRules;
      for (var j = 0; j < rules.length; j++) {
        var rule = rules[j];
        if (typeof(rule.style) != "undefined") {
          var elems = dom.querySelectorAll(rule.selectorText);
          if (elems.length > 0) {
            used += rule.selectorText + " { " + rule.style.cssText + " }\n";
          }
        }
      }
    }catch(err){
      // pass
    }
  }
  var s = document.createElement('style');
  s.setAttribute('type', 'text/css');
  s.innerHTML = "<![CDATA[\n" + used + "\n]]>";
 
  var defs = document.createElement('defs');
  defs.appendChild(s);
  dom.insertBefore(defs, dom.firstChild);
}


function setAttributes(el) {
  el.setAttribute("version", "1.1");
  el.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  el.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink");
}


function outerHTML(el) {
  var outer = document.createElement('div');
  outer.appendChild(el.cloneNode(true));
  return outer.innerHTML;
}


function svg2image(xml) {
  var image = new Image();
  image.src = 'data:image/svg+xml;base64,' + btoa(xml);
  return image;
}


function downloadPNG(){
  let svg = d3.select('#graph');
  styles(svg.node());
  setAttributes(svg.node());
  let xml = outerHTML(svg.node());
  let image = svg2image(xml);

  let canvas = d3.select('body').append('canvas');
  let bbox = d3.select('#graph').node().getBoundingClientRect();
  canvas.node().width = bbox.width;
  canvas.node().height = bbox.height;
  // canvas.node().width = 500;
  // canvas.node().height = 500;


  var context = canvas.node().getContext('2d');

  setTimeout(()=>{
    context.drawImage(image, 0, 0);
    var a = d3.select('body').append('a');
    a.node().download = "image.png";
    a.node().href = canvas.node().toDataURL('image/png');
    a.node().click();

    canvas.remove();
    a.remove();
  }, 10);
  return [context, image];
}



//// export JSON
function exportJson(obj, fn='result.json'){
  let objStr = JSON.stringify(obj, null, 2);
  let dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(objStr);
  var anchor = document.getElementById('download-json');
  anchor.setAttribute('href', dataStr);
  anchor.setAttribute('download', fn);
  anchor.click();
}