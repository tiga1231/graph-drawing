// main
let isPlaying = true;
window.onload = function(){
  let graphTypeSelect = d3.select('#graphType');
  let fn = `data/${graphTypeSelect.node().value}.json`;

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

  let vertexResolutionSlider = d3.select('#vertexResolutionSlider');
  let vertexResolutionLabel = d3.select('#vertexResolutionLabel');

  let gabrielSlider = d3.select('#gabrielSlider');
  let gabrielLabel = d3.select('#gabrielLabel');
  let aspectRatioSlider = d3.select('#aspectRatioSlider');
  let aspectRatioLabel = d3.select('#aspectRatioLabel');

  // let areaSlider = d3.select('#areaSlider');
  // let areaLabel = d3.select('#areaLabel');
  // let desiredAreaSlider = d3.select('#desiredAreaSlider');
  // let desiredAreaLabel = d3.select('#desiredAreaLabel');


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
    // area: (areaSlider.node() !== null ? +areaSlider.node().value : 0.0),
  };

  let sampleSize = 5;
 

  let maxPlotIter = 50;
  let niter = 20000;
  let maxMetricSize = 10;

  let optimizers = [tf.train.momentum(lr, momentum, false)];
  window.optimizers = optimizers;
  let losses = [];
  let metrics = [];
  window.metrics = metrics;
  let animId = 0;

  let svg_loss = d3.select('#loss');
  let svg_graph = d3.select('#graph');

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


  function loadGraph(fn){
    d3.json(fn).then((graph)=>{
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
          // console.log(record);
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
          window.metrics = metrics;
          if (losses.length >= 10){
            let n = losses.length;
            let firstSlice = losses.slice(Math.floor(n/2), Math.floor(n/4*3));
            let secondSlice = losses.slice(Math.floor(n/4*3), n);
            let avgLoss0 = math.mean(firstSlice);
            let avgLoss1 = math.mean(secondSlice);
            if(avgLoss1 > avgLoss0){
              lrSlider.on('input')(Math.max(lr/1.01, 0.001 ));
            }
          }
          updateNodePosition(graph, x.arraySync());
          drawGraph(svg_graph, graph);
        });
      }
      play();
      
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
        reset();
        svg_graph.xDomain = undefined;
        let xy = x.arraySync();
        graph.nodes.forEach((d,i)=>{
          d.x = xy[i][0];
          d.x = xy[i][1];
        });
        dataObj.x = x;
        drawGraph(svg_graph, graph);
      });

      downloadButton.on('click', ()=>{
        let res = {};
        res.nodes = graph.nodes;
        res.edges = graph.edges.map(d=>({
          source: d.source.id, 
          target: d.target.id,
        }));
        res.coef = coef;
        res.optimizer = {
          lr: optimizers[0].learningRate, 
          momentum: optimizers[0].momentum
        };
        res.graphDistance = graph.graphDistance;
        res.weight = graph.weight;
        let res_json = JSON.stringify(res, null, 2);
        let file_name = `${graphTypeSelect.node().value}.json`;
        let dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(res_json);
        var anchor = document.getElementById('download-json');
        anchor.setAttribute('href', dataStr);
        anchor.setAttribute('download', file_name);
        anchor.click();
      });

      graphTypeSelect.on('change', function(){
        let fn = d3.select(this).node().value;
        fn = `data/${fn}.json`;
        cancelAnimationFrame(dataObj.animId);
        loadGraph(fn);
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

      // areaSlider.on('input', function(){
      //   let value = +areaSlider.node().value;
      //   areaLabel.text(value.toFixed(2));
      //   coef['area'] = value;
      // });
      // areaSlider.on('input')();
      // desiredAreaSlider.on('input', function(){
      //   let value = +desiredAreaSlider.node().value;
      //   desiredAreaLabel.text(value.toFixed(2));
      //   dataObj['desiredArea'] = value;
      // });
      // desiredAreaSlider.on('input')();


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



    });
  }//loadGraph end
  loadGraph(fn);
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
  // image.src = 'data:image/svg+xml;base64,' + btoa('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" id="Layer_1" x="0px" y="0px" viewBox="0 0 100 100" enable-background="new 0 0 100 100" xml:space="preserve" height="100px" width="100px"><g><path d="M28.1,36.6c4.6,1.9,12.2,1.6,20.9,1.1c8.9-0.4,19-0.9,28.9,0.9c6.3,1.2,11.9,3.1,16.8,6c-1.5-12.2-7.9-23.7-18.6-31.3   c-4.9-0.2-9.9,0.3-14.8,1.4C47.8,17.9,36.2,25.6,28.1,36.6z"/><path d="M70.3,9.8C57.5,3.4,42.8,3.6,30.5,9.5c-3,6-8.4,19.6-5.3,24.9c8.6-11.7,20.9-19.8,35.2-23.1C63.7,10.5,67,10,70.3,9.8z"/><path d="M16.5,51.3c0.6-1.7,1.2-3.4,2-5.1c-3.8-3.4-7.5-7-11-10.8c-2.1,6.1-2.8,12.5-2.3,18.7C9.6,51.1,13.4,50.2,16.5,51.3z"/><path d="M9,31.6c3.5,3.9,7.2,7.6,11.1,11.1c0.8-1.6,1.7-3.1,2.6-4.6c0.1-0.2,0.3-0.4,0.4-0.6c-2.9-3.3-3.1-9.2-0.6-17.6   c0.8-2.7,1.8-5.3,2.7-7.4c-5.2,3.4-9.8,8-13.3,13.7C10.8,27.9,9.8,29.7,9,31.6z"/><path d="M15.4,54.7c-2.6-1-6.1,0.7-9.7,3.4c1.2,6.6,3.9,13,8,18.5C13,69.3,13.5,61.8,15.4,54.7z"/><path d="M39.8,57.6C54.3,66.7,70,73,86.5,76.4c0.6-0.8,1.1-1.6,1.7-2.5c4.8-7.7,7-16.3,6.8-24.8c-13.8-9.3-31.3-8.4-45.8-7.7   c-9.5,0.5-17.8,0.9-23.2-1.7c-0.1,0.1-0.2,0.3-0.3,0.4c-1,1.7-2,3.4-2.9,5.1C28.2,49.7,33.8,53.9,39.8,57.6z"/><path d="M26.2,88.2c3.3,2,6.7,3.6,10.2,4.7c-3.5-6.2-6.3-12.6-8.8-18.5c-3.1-7.2-5.8-13.5-9-17.2c-1.9,8-2,16.4-0.3,24.7   C20.6,84.2,23.2,86.3,26.2,88.2z"/><path d="M30.9,73c2.9,6.8,6.1,14.4,10.5,21.2c15.6,3,32-2.3,42.6-14.6C67.7,76,52.2,69.6,37.9,60.7C32,57,26.5,53,21.3,48.6   c-0.6,1.5-1.2,3-1.7,4.6C24.1,57.1,27.3,64.5,30.9,73z"/></g></svg>');
  return image;
}

function downloadSvg(){
  let svg = d3.select('#graph');
  styles(svg.node());
  setAttributes(svg.node());
  let xml = outerHTML(svg.node());
  let image = svg2image(xml);

  var canvas = d3.select('body').append('canvas');
  // canvas.node().width = image.width;
  // canvas.node().height = image.height;
  canvas.node().width = 500;
  canvas.node().height = 500;


  var context = canvas.node().getContext('2d');

  setTimeout(()=>{
    context.drawImage(image, 0, 0);
  }, 1000);
  

  // console.log(canvas.node().toDataURL('image/png'));
  // var a = d3.select('body').append('a');
  // a.node().download = "image.png";
  // a.node().href = canvas.node().toDataURL('image/png');
  // a.node().click();

  return [context, image];
}


